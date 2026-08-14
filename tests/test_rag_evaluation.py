"""Evaluates the RAG pipeline's actual answer quality against a golden
dataset - a different kind of check than the rest of the test suite.
Everywhere else, tests check code correctness (deterministic, pass/fail,
mostly against fake LLM responses so they aren't flaky). This file checks
whether the *real* model's output is actually good - grounded, relevant,
not hallucinated - which is inherently statistical, not pass/fail.

Deliberately not living inside app/: this is a test of the production RAG
code (DocumentSearchTool + LlmGateway.answer_from_context, the exact
functions app/graph/agent_graph.py calls), not a reimplementation of it.

Opt-in, like the other live tests: requires a real GROQ_API_KEY, since
every question here costs several real Groq calls (one for synthesis,
several more for the LLM-as-judge metrics).

Run with: uv run pytest tests/test_rag_evaluation.py -v -s
(-s so the per-question score printout is visible)
"""

import asyncio

import pytest
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.graph.agent_graph import _format_context
from app.services.document_index import DocumentIndex
from app.services.llm_gateway import LlmGateway
from app.tools.document_search_tool import DocumentSearchInput, DocumentSearchTool

pytestmark = pytest.mark.skipif(
    not get_settings().groq_api_key, reason="requires a real GROQ_API_KEY in .env"
)

# Each case has a `reference` (ground truth) so both retrieval quality
# (precision/recall) and answer quality (faithfulness/relevancy) can be
# scored against something concrete, not just checked for plausibility.
_ANSWERABLE_CASES = [
    {
        "question": "What are the password requirements?",
        "reference": (
            "Passwords must be at least 12 characters, include a mix of letters, "
            "numbers, and symbols, and be rotated every 90 days. Multi-factor "
            "authentication is required for accounts with access to PHI."
        ),
    },
    {
        "question": "How much paid time off do full-time employees get?",
        "reference": (
            "Full-time employees accrue 15 days of PTO per year, plus 10 company holidays."
        ),
    },
    {
        "question": "How long does provider onboarding typically take?",
        "reference": (
            "Provider onboarding typically takes two to four weeks, depending on "
            "the provider's size and existing systems."
        ),
    },
    {
        "question": "How long are patient records retained?",
        "reference": (
            "Patient records are retained for seven years after the last date of "
            "service, or longer if required by the provider's state regulations."
        ),
    },
    {
        "question": "What is the response time SLA for an urgent-priority ticket?",
        "reference": "Urgent-priority tickets receive an initial response within one hour.",
    },
    {
        "question": "Can employees use their personal laptop to access patient data?",
        "reference": (
            "No. Personal devices may not be used to access the MedFlow platform "
            "under any circumstances - a company-issued laptop must be used for "
            "all work involving PHI or provider data."
        ),
    },
    {
        "question": "What's the first thing to do if you suspect a security incident?",
        "reference": (
            "Report the suspected incident to the Security team within one hour of "
            "discovery, using the incident hotline or the #security-incidents Slack channel."
        ),
    },
    # Deliberately ambiguous: both hipaa_security_policy.md (one line) and
    # security_incident_runbook.md (a full four-step process) touch this -
    # a real test of whether retrieval finds the actually-useful document,
    # not just *a* document that mentions the topic.
    {
        "question": "Who do I report a suspected data breach to, and how quickly?",
        "reference": "Report to the Security team within one hour of discovery.",
    },
    {
        "question": "What happens to a provider's data if MedFlow receives a deletion request?",
        "reference": (
            "Deletion requests are processed within 30 days and require sign-off "
            "from both the Compliance Officer and the provider's administrator, "
            "except for patient records still within their mandatory retention period."
        ),
    },
]

# No reference here on purpose - these questions aren't answered anywhere
# in the corpus. The thing being tested is whether the system correctly
# declines instead of guessing, which is the actual hallucination/
# grounding check the retrieval-only metrics above can't catch (they'd
# happily score a confident, plausible-sounding, wrong answer as relevant).
_UNANSWERABLE_QUESTIONS = [
    "Can employees carry over unused PTO to the next year?",
    "What is the maximum file size for a support ticket attachment?",
    "Does MedFlow offer a signing bonus for new engineering hires?",
]

def _judge_declined_to_answer(question: str, answer: str) -> bool:
    # LLM-as-judge, not a fixed phrase list: real refusals get phrased too
    # many different ways across calls ("I don't have enough information,"
    # "I don't have that information," "That isn't covered in our
    # documents," ...) for exact/substring matching to survive normal LLM
    # phrasing variance without turning into permanent whack-a-mole.
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nAnswer: {answer}\n\n"
                    "Does the answer decline to answer (e.g. says it doesn't have "
                    "enough information), or does it confidently assert a specific "
                    "answer to the question? Respond with exactly one word: "
                    "DECLINED or ANSWERED."
                ),
            }
        ],
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "DECLINED" in verdict


def _judge_llm() -> LangchainLLMWrapper:
    # Groq's API is OpenAI-compatible, so langchain_openai's client works
    # against it directly via base_url - avoids langchain-groq, which pins
    # groq<1.0.0 and conflicts with this project's groq>=1.6.0 dependency.
    settings = get_settings()
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    )


def _judge_embeddings() -> LangchainEmbeddingsWrapper:
    # Local, not an API call - same model DocumentIndex already uses.
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


def _real_gateway(db_session: Session) -> LlmGateway:
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)
    return LlmGateway(client, settings.groq_model, db_session, feature="rag_eval")


def test_rag_answer_quality_against_golden_dataset(
    document_index: DocumentIndex, db_session: Session
) -> None:
    document_search_tool = DocumentSearchTool(document_index)
    gateway = _real_gateway(db_session)

    judge_llm = _judge_llm()
    faithfulness = Faithfulness(llm=judge_llm)
    # strictness=1, not the default 3: AnswerRelevancy asks the judge LLM
    # for multiple candidate reverse-engineered questions in one call via
    # the `n` parameter: Groq's API rejects n>1 ("number must be at most
    # 1"), unlike OpenAI's, which the default was tuned against.
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=_judge_embeddings(), strictness=1)
    context_precision = LLMContextPrecisionWithReference(llm=judge_llm)
    context_recall = LLMContextRecall(llm=judge_llm)

    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "relevancy": [],
        "precision": [],
        "recall": [],
    }

    for case in _ANSWERABLE_CASES:
        result = document_search_tool.run(DocumentSearchInput(query=case["question"]))
        assert result.found, f"retrieval found nothing for: {case['question']}"
        contexts = [chunk.content for chunk in result.chunks]

        synthesized = gateway.answer_from_context(case["question"], _format_context(result.chunks))
        assert not synthesized.blocked

        sample = SingleTurnSample(
            user_input=case["question"],
            response=synthesized.text,
            retrieved_contexts=contexts,
            reference=case["reference"],
        )

        f = asyncio.run(faithfulness.single_turn_ascore(sample))
        r = asyncio.run(answer_relevancy.single_turn_ascore(sample))
        p = asyncio.run(context_precision.single_turn_ascore(sample))
        rc = asyncio.run(context_recall.single_turn_ascore(sample))

        print(
            f"\n[{case['question']}]\n"
            f"  faithfulness={f:.2f} relevancy={r:.2f} precision={p:.2f} recall={rc:.2f}\n"
            f"  answer: {synthesized.text}"
        )
        scores["faithfulness"].append(f)
        scores["relevancy"].append(r)
        scores["precision"].append(p)
        scores["recall"].append(rc)

    averages = {name: sum(values) / len(values) for name, values in scores.items()}
    print(f"\nAverages across {len(_ANSWERABLE_CASES)} questions: {averages}")

    # Thresholds are a floor for catching real regressions (a prompt or
    # chunking change that quietly degrades quality), not a claim that
    # these are the "right" scores - calibrated below what this pipeline
    # actually scores today, with margin for normal LLM-judge variance.
    assert averages["faithfulness"] >= 0.8, "answers include facts not present in retrieved context"
    assert averages["relevancy"] >= 0.6, "answers are drifting from what was actually asked"
    assert averages["precision"] >= 0.5, "retrieval is ranking irrelevant chunks too highly"
    assert averages["recall"] >= 0.5, "retrieval is missing information the reference expects"


def test_rag_declines_unanswerable_questions_instead_of_guessing(
    document_index: DocumentIndex, db_session: Session
) -> None:
    document_search_tool = DocumentSearchTool(document_index)
    gateway = _real_gateway(db_session)

    for question in _UNANSWERABLE_QUESTIONS:
        result = document_search_tool.run(DocumentSearchInput(query=question))
        # Retrieval always returns its top_k=3 nearest chunks regardless of
        # relevance (no similarity threshold) - so `found` stays True even
        # here. The actual grounding behavior under test is downstream, in
        # synthesis: does it recognize these chunks don't answer the
        # question and decline, or does it hallucinate a confident answer
        # anyway?
        synthesized = gateway.answer_from_context(question, _format_context(result.chunks))
        assert not synthesized.blocked

        print(f"\n[{question}]\n  answer: {synthesized.text}")
        assert _judge_declined_to_answer(question, synthesized.text), (
            f"expected a refusal for an unanswerable question, got: {synthesized.text!r}"
        )
