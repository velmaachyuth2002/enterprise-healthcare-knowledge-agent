from collections.abc import Callable
from datetime import date
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tools.policy_tool import PolicyLookupInput, PolicyLookupResult, PolicyTool
from app.tools.sql_tool import (
    TicketCountInput,
    TicketCountResult,
    TicketCountTool,
    UnresolvedTicketsResult,
    UnresolvedTicketsTool,
)

ToolResult = PolicyLookupResult | TicketCountResult | UnresolvedTicketsResult

Decision = Literal[
    "call_policy_tool",
    "call_ticket_count_tool",
    "call_unresolved_tickets_tool",
    "respond",
]


class AgentState(TypedDict):
    question: str
    decision: Decision | None
    topic: str | None
    count_range_start: date | None
    count_range_end: date | None
    tool_result: ToolResult | None
    answer: str | None


def _extract_topic(question: str, known_topics: list[str]) -> str | None:
    normalized = question.lower()
    return next((topic for topic in known_topics if topic.replace("_", " ") in normalized), None)


def _last_month_range(today: date) -> tuple[date, date]:
    end = today.replace(day=1)
    if end.month == 1:
        start = end.replace(year=end.year - 1, month=12)
    else:
        start = end.replace(month=end.month - 1)
    return start, end


def _route_after_planner(state: AgentState) -> Decision:
    # `_planner` always runs immediately before this and always sets
    # `decision`, so no fallback is needed here.
    return state["decision"]


def _respond(state: AgentState) -> dict[str, str]:
    result = state.get("tool_result")
    if result is None:
        return {"answer": "I don't have a way to answer that yet."}

    if isinstance(result, PolicyLookupResult):
        if not result.found:
            return {"answer": "I couldn't find a policy on that topic."}
        return {"answer": f"{result.policy_name}: {result.content}"}

    if isinstance(result, TicketCountResult):
        return {"answer": f"{result.count} ticket(s) were opened in that period."}

    if isinstance(result, UnresolvedTicketsResult):
        if not result.tickets:
            return {"answer": "There are no unresolved tickets."}
        subjects = "; ".join(ticket.subject for ticket in result.tickets)
        return {"answer": f"{len(result.tickets)} unresolved ticket(s): {subjects}"}

    # Not a defensive no-op: unlike the state-field guarantees above (which
    # our own control flow makes provably true), "every ToolResult member is
    # handled" is a property that rots the moment a new tool is added and
    # this function isn't updated to match. Fail loudly at the gap instead
    # of a KeyError three layers away when `answer` turns out unset.
    raise AssertionError(f"unhandled tool result type: {type(result)}")


def build_graph(
    policy_tool: PolicyTool,
    ticket_count_tool: TicketCountTool,
    unresolved_tickets_tool: UnresolvedTicketsTool,
    today: Callable[[], date] = date.today,
):
    known_topics = policy_tool.known_topics()

    def _planner(state: AgentState) -> dict[str, object]:
        # Deterministic stand-in for an LLM-driven planner: matches
        # keywords instead of true intent understanding. Everything
        # downstream only reads `decision` and whatever arguments were set
        # alongside it, so swapping this for a real LLM call later touches
        # this function only.
        #
        # Known limitation: any ticket-count-shaped question always
        # resolves to "last month", regardless of what range it actually
        # asked for - fixed once a real LLM planner can parse the relative
        # date expression itself.
        question = state["question"].lower()

        topic = _extract_topic(question, known_topics)
        if topic is not None:
            return {"decision": "call_policy_tool", "topic": topic}

        if "unresolved" in question:
            return {"decision": "call_unresolved_tickets_tool"}

        if "ticket" in question and ("how many" in question or "count" in question):
            start, end = _last_month_range(today())
            return {
                "decision": "call_ticket_count_tool",
                "count_range_start": start,
                "count_range_end": end,
            }

        if "policy" in question:
            # Looks policy-shaped but didn't match a known topic - still
            # try the tool so it can report "not found" explicitly, rather
            # than silently falling back to the generic no-tool answer.
            return {"decision": "call_policy_tool", "topic": state["question"]}

        return {"decision": "respond"}

    def _call_policy_tool(state: AgentState) -> dict[str, PolicyLookupResult]:
        # `_planner` only routes here when it has also set `topic`.
        result = policy_tool.run(PolicyLookupInput(topic=state["topic"]))
        return {"tool_result": result}

    def _call_ticket_count_tool(state: AgentState) -> dict[str, TicketCountResult]:
        # `_planner` only routes here when it has also set both range fields.
        result = ticket_count_tool.run(
            TicketCountInput(start=state["count_range_start"], end=state["count_range_end"])
        )
        return {"tool_result": result}

    def _call_unresolved_tickets_tool(state: AgentState) -> dict[str, UnresolvedTicketsResult]:
        result = unresolved_tickets_tool.run()
        return {"tool_result": result}

    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("call_policy_tool", _call_policy_tool)
    graph.add_node("call_ticket_count_tool", _call_ticket_count_tool)
    graph.add_node("call_unresolved_tickets_tool", _call_unresolved_tickets_tool)
    graph.add_node("respond", _respond)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "call_policy_tool": "call_policy_tool",
            "call_ticket_count_tool": "call_ticket_count_tool",
            "call_unresolved_tickets_tool": "call_unresolved_tickets_tool",
            "respond": "respond",
        },
    )
    graph.add_edge("call_policy_tool", "respond")
    graph.add_edge("call_ticket_count_tool", "respond")
    graph.add_edge("call_unresolved_tickets_tool", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
