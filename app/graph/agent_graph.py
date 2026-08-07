from collections.abc import Callable
from datetime import date
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.services.llm_gateway import LlmGateway, ToolSpec
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

_CLARIFY_MESSAGE = (
    "I understood what you wanted, but couldn't quite parse the details - could you rephrase?"
)


class AgentState(TypedDict):
    question: str
    decision: Decision | None
    blocked: bool | None
    planner_content: str | None
    topic: str | None
    count_range_start: date | None
    count_range_end: date | None
    tool_result: ToolResult | None
    answer: str | None


def _route_after_planner(state: AgentState) -> Decision:
    # `_planner` always runs immediately before this and always sets
    # `decision`, so no fallback is needed here.
    return state["decision"]


def _respond(state: AgentState) -> dict[str, str]:
    if state.get("blocked"):
        return {"answer": "I can't help with that request."}

    result = state.get("tool_result")
    if result is None:
        planner_content = state.get("planner_content")
        if planner_content:
            return {"answer": planner_content}
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
    gateway: LlmGateway,
    today: Callable[[], date] = date.today,
):
    # Built once, not per-invocation. PolicyTool's description is enriched
    # with its real known topics - without this the model has no way to
    # know what a valid `topic` argument looks like and will hallucinate
    # one that never matches anything in the tool's lookup table.
    tool_specs = [
        ToolSpec(
            name=policy_tool.name,
            description=(
                f"{policy_tool.description} "
                f"Valid topics: {', '.join(policy_tool.known_topics())}."
            ),
            parameters=PolicyLookupInput.model_json_schema(),
        ),
        ToolSpec(
            name=ticket_count_tool.name,
            description=ticket_count_tool.description,
            parameters=TicketCountInput.model_json_schema(),
        ),
        ToolSpec(
            name=unresolved_tickets_tool.name,
            description=unresolved_tickets_tool.description,
            parameters={"type": "object", "properties": {}},
        ),
    ]

    def _planner(state: AgentState) -> dict[str, object]:
        # `today()` is injected (not date.today() called directly) for two
        # reasons: testability, and because the model has no built-in
        # notion of the current date - without this, relative expressions
        # like "last month" have no principled way to resolve.
        question = f"Today's date is {today().isoformat()}. {state['question']}"
        decision = gateway.decide_tool_call(question, tool_specs)

        if decision.blocked:
            return {"decision": "respond", "blocked": True}

        if decision.tool_name == policy_tool.name:
            try:
                args = PolicyLookupInput.model_validate(decision.arguments)
            except ValidationError:
                return {"decision": "respond", "planner_content": _CLARIFY_MESSAGE}
            return {"decision": "call_policy_tool", "topic": args.topic}

        if decision.tool_name == ticket_count_tool.name:
            try:
                args = TicketCountInput.model_validate(decision.arguments)
            except ValidationError:
                return {"decision": "respond", "planner_content": _CLARIFY_MESSAGE}
            return {
                "decision": "call_ticket_count_tool",
                "count_range_start": args.start,
                "count_range_end": args.end,
            }

        if decision.tool_name == unresolved_tickets_tool.name:
            return {"decision": "call_unresolved_tickets_tool"}

        # decision.tool_name is either None (model chose to just respond)
        # or names a tool that doesn't exist (hallucinated) - both are
        # handled the same way: nothing to execute, fall back to whatever
        # plain-text reply the model gave.
        return {"decision": "respond", "planner_content": decision.content}

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
