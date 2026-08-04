from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tools.policy_tool import PolicyLookupInput, PolicyLookupResult, PolicyTool


class AgentState(TypedDict):
    question: str
    decision: Literal["call_policy_tool", "respond"] | None
    topic: str | None
    tool_result: PolicyLookupResult | None
    answer: str | None


def _extract_topic(question: str, known_topics: list[str]) -> str | None:
    normalized = question.lower()
    return next((topic for topic in known_topics if topic.replace("_", " ") in normalized), None)


def _route_after_planner(state: AgentState) -> Literal["call_policy_tool", "respond"]:
    # `_planner` always runs immediately before this and always sets `decision`,
    # so no fallback is needed here — nothing else populates this edge.
    return state["decision"]


def _respond(state: AgentState) -> dict[str, str]:
    result = state.get("tool_result")
    if result is None:
        return {"answer": "I don't have a way to answer that yet."}
    if not result.found:
        return {"answer": "I couldn't find a policy on that topic."}
    return {"answer": f"{result.policy_name}: {result.content}"}


def build_graph(policy_tool: PolicyTool):
    known_topics = policy_tool.known_topics()

    def _planner(state: AgentState) -> dict[str, str | None]:
        # Deterministic stand-in for an LLM-driven planner: matches the
        # question against the tool's known topics instead of true intent
        # understanding. Everything downstream only reads `decision`/`topic`
        # from state, so swapping this for a real LLM call later touches
        # this function only.
        topic = _extract_topic(state["question"], known_topics)
        if topic is not None:
            return {"decision": "call_policy_tool", "topic": topic}
        if "policy" in state["question"].lower():
            # Looks policy-shaped but didn't match a known topic — still try
            # the tool so it can report "not found" explicitly, rather than
            # silently falling back to the generic no-tool answer.
            return {"decision": "call_policy_tool", "topic": state["question"]}
        return {"decision": "respond", "topic": None}

    def _call_policy_tool(state: AgentState) -> dict[str, PolicyLookupResult]:
        # `_planner` only routes here when it has also set `topic`, so no
        # None-handling is needed for that field at this point.
        result = policy_tool.run(PolicyLookupInput(topic=state["topic"]))
        return {"tool_result": result}

    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("call_policy_tool", _call_policy_tool)
    graph.add_node("respond", _respond)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"call_policy_tool": "call_policy_tool", "respond": "respond"},
    )
    graph.add_edge("call_policy_tool", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
