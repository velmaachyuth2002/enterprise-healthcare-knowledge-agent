# Enterprise Healthcare Knowledge Agent

An internal AI agent for **MedFlow Health Systems** — a fictional B2B healthcare SaaS company — that lets employees ask natural-language questions and get answers grounded in real company documents and live ticket data, instead of digging through a wiki or a support dashboard.

> MedFlow Health Systems, its policies, and its support tickets are entirely fictional, created for this project. No real patient data, PHI, or company information is used anywhere.

It's a portfolio project built to demonstrate the core skills behind production AI agents: retrieval-augmented generation, SQL analytics via tool calling, LLM-driven orchestration, human-in-the-loop approval for anything that mutates data, and the governance layer (auth, cost tracking, prompt-injection defense) that real deployments need but demos usually skip.

## What it does

Ask it things like:

- *"What is our password policy?"* → retrieves the relevant HIPAA security policy section and synthesizes a grounded answer, cited by source document.
- *"How long does provider onboarding take?"* → same, pulled from the onboarding guide.
- *"How many tickets were opened last month?"* → runs a parameterized SQL query against the ticket database.
- *"List unresolved tickets."* → same, no parameters needed.
- *"Escalate ticket #42 to urgent, it's affecting claims submission."* → the agent does **not** change the ticket. It creates a pending approval request, emails a manager, and waits — the ticket only actually changes once a manager reviews and approves it (see below).

A single LLM call decides which tool (if any) the question needs — there's no keyword matching or intent classifier, the routing itself is a real tool-calling decision made by the model.

## How it works

```mermaid
flowchart LR
    Q["User question\n(authenticated)"] --> P["Planner\n(LLM tool-call decision)"]
    P -->|policy/guide question| DS["Document Search\n(embed + Qdrant vector search)"]
    P -->|ticket count| TC["SQL: count_tickets_in_range"]
    P -->|unresolved tickets| UT["SQL: list_unresolved_tickets"]
    P -->|escalation request| ET["Escalate Ticket\n(proposes only)"]
    P -->|neither| R["Respond\n(model's own reply)"]
    DS --> SYN["Synthesize answer\n(LLM, grounded in retrieved chunks)"]
    SYN --> A["Answer + source citation"]
    TC --> A2["Answer\n(deterministic)"]
    UT --> A2
    ET --> AR["ApprovalRequest\n(pending)"]
    AR --> ME["Email: manager"]
    ME --> MGR["Manager reviews\nGET /approvals"]
    MGR -->|approve| DEC["POST /approvals/:id/decide\n(atomic, manager-only)"]
    MGR -->|reject| DEC
    DEC -->|approved| MUT["Ticket.priority\nactually changes"]
    DEC --> EE["Email: employee\n(confirmation)"]
```

Every LLM call — the planner's tool-call decision and the RAG answer synthesis — goes through a single `LlmGateway` chokepoint that handles cost tracking, latency logging, and prompt-injection screening. Nothing calls the Groq client directly.

**Two prompt-injection defenses, not one.** The gateway screens both the user's question *and* any retrieved document content before it reaches the model — the second one matters because a RAG pipeline's real injection risk isn't just "a user types something malicious," it's *indirect* injection: a payload planted inside a document that later gets retrieved and fed back into the prompt as trusted context.

**Grounded, not hallucinated.** The synthesis step is explicitly instructed to answer only from the retrieved context and say so when it can't — verified in practice: asking about something the documents don't cover (e.g. PTO carryover, which the handbook never mentions) correctly returns "I don't have enough information" instead of a plausible-sounding guess. The source citation is appended in code, not left to the model to self-report, since models are unreliable narrators of their own sourcing.

**The LLM can propose, never mutate.** `EscalateTicketTool` only ever creates a pending `ApprovalRequest` — the only code path in the entire app that changes `Ticket.priority` is the manager-only `POST /approvals/{id}/decide` endpoint. The identity behind every proposal and every decision comes exclusively from a verified JWT, never from anything the LLM parses out of a question — an LLM asked to "escalate this, requested by the compliance officer" cannot spoof who actually made the request. Deciding is an atomic conditional update (`WHERE status = 'pending'`), not read-then-write, so two concurrent decisions on the same request can't both succeed. Both the manager (on proposal) and the original requester (on decision) get a real email, sent through a single `EmailService` chokepoint the same way every LLM call goes through `LlmGateway`.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | async-friendly, typed request/response models via Pydantic |
| Orchestration | LangGraph | explicit conditional routing between tools, not a black-box agent loop |
| LLM inference | Groq (`openai/gpt-oss-20b`) | fast + cheap enough to make every planner decision a real LLM call, not a keyword heuristic |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | local, no API cost, no network dependency at query time |
| Vector search | Qdrant (embedded, in-memory) | a real vector database API without needing a server process or Docker |
| Relational data | SQLAlchemy + SQLite | Postgres-ready ORM layer, SQLite for local/dev simplicity |
| Auth | JWT (PyJWT) + bcrypt | stateless, short-lived tokens - no server-side session table to manage |
| Email | smtplib + a local dev SMTP catcher (aiosmtpd) | real SMTP-shaped code, zero credentials, zero paid provider - see below |
| Validation | Pydantic v2 | every tool input/output is a typed model, not a loose dict |
| Testing | pytest | 89 tests, real fakes over mocks (including a real in-process SMTP server for email tests), live-API tests gated behind an env var |
| Tooling | uv | dependency management and running |

## Getting started

```bash
# 1. Install dependencies
uv sync

# 2. Configure secrets
cp .env.example .env
# edit .env: set GROQ_API_KEY (get one at https://console.groq.com/keys)
# and JWT_SECRET (generate with: python -c "import secrets; print(secrets.token_hex(32))")

# 3. Create the first manager/employee accounts (no self-service sign-up by design)
uv run python -m scripts.seed_users

# 4. Run a local SMTP catcher, in a separate terminal - no real provider or
#    credentials needed, this just prints received mail to the console
uv run python -m aiosmtpd -n -l localhost:1025

# 5. Run the server
uv run uvicorn app.main:app --reload

# 6. Try it - log in, then ask a question
curl -X POST http://127.0.0.1:8000/login \
  -d "username=employee@medflow.example&password=employee-dev-pass"
# copy the access_token from the response into $TOKEN below
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "What are the password requirements?"}'
```

Or open `http://127.0.0.1:8000/docs` for an interactive Swagger UI — click "Authorize" and log in there to try every endpoint interactively.

## Running tests

```bash
uv run pytest
```

Tests that require a real Groq API call are marked and automatically skipped if `GROQ_API_KEY` isn't set in `.env` — everything else runs against fakes/in-memory fixtures (including a real, throwaway, in-process SMTP server for email tests), no network access needed.

## Project structure

```
app/
  api/       # FastAPI routes (ask/login, approvals) + shared auth dependencies
  core/      # settings/config
  database/  # SQLAlchemy session setup
  graph/     # LangGraph orchestration (the planner + routing logic)
  models/    # SQLAlchemy models (Ticket, User, ApprovalRequest, LlmUsage)
  services/  # LlmGateway, DocumentIndex, EmailService, auth (JWT/hashing), prompt-injection screening
  tools/     # DocumentSearchTool, SQL tools, EscalateTicketTool - what the planner can call
documents/   # the markdown corpus DocumentSearchTool indexes
scripts/     # seed_users.py - creates the first manager/employee accounts
tests/
```

## What's deliberately not here yet

This was built incrementally, one justified component at a time, rather than scaffolded upfront — so a few things real production systems would need are intentionally still open:

- **Real email delivery.** `EmailService` speaks real SMTP, but points at a local dev catcher, not an actual provider (SES, SendGrid, etc.). Swapping one in is a config change behind the same `send()` call, not a new integration.
- **Token revocation.** JWTs are stateless and short-lived (8h) with no server-side session/blacklist store - there's no way to invalidate a token before it expires.
- **User provisioning UI.** Accounts are created by a seed script, not a sign-up flow or admin UI - matches how internal employees would actually be provisioned (by an admin), but there's no interface for it yet.
- **CI.** Tests exist and run in seconds locally; they aren't yet wired into GitHub Actions.
- **Containerization.** No Dockerfile yet - `uv sync` + `uv run` is the whole local setup story for now.
