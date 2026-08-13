from collections.abc import Callable
from datetime import date
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.services.document_index import ScoredChunk
from app.services.email_service import EmailService
from app.services.llm_gateway import LlmGateway, ToolSpec
from app.tools.document_search_tool import (
    DocumentSearchInput,
    DocumentSearchResult,
    DocumentSearchTool,
)
from app.tools.escalate_ticket_tool import (
    EscalateTicketInput,
    EscalateTicketResult,
    EscalateTicketTool,
)
from app.tools.sql_tool import (
    TicketCountInput,
    TicketCountResult,
    TicketCountTool,
    UnresolvedTicketsResult,
    UnresolvedTicketsTool,
)

ToolResult = (
    DocumentSearchResult | TicketCountResult | UnresolvedTicketsResult | EscalateTicketResult
)

Decision = Literal[
    "call_document_search_tool",
    "call_ticket_count_tool",
    "call_unresolved_tickets_tool",
    "call_escalate_ticket_tool",
    "respond",
]

_CLARIFY_MESSAGE = (
    "I understood what you wanted, but couldn't quite parse the details - could you rephrase?"
)


class AgentState(TypedDict):
    question: str
    # Set by the API layer from the authenticated JWT - never by the
    # planner or anything parsed from the LLM's tool-call arguments.
    requester_id: int | None
    decision: Decision | None
    blocked: bool | None
    planner_content: str | None
    query: str | None
    count_range_start: date | None
    count_range_end: date | None
    ticket_id: int | None
    priority: str | None
    reason: str | None
    tool_result: ToolResult | None
    answer: str | None


def _route_after_planner(state: AgentState) -> Decision:
    # `_planner` always runs immediately before this and always sets
    # `decision`, so no fallback is needed here.
    return state["decision"]


def _format_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[{chunk.source} — {chunk.heading}]\n{chunk.content}" for chunk in chunks)


def build_graph(
    document_search_tool: DocumentSearchTool,
    ticket_count_tool: TicketCountTool,
    unresolved_tickets_tool: UnresolvedTicketsTool,
    escalate_ticket_tool: EscalateTicketTool,
    email_service: EmailService,
    get_manager_emails: Callable[[], list[str]],
    gateway: LlmGateway,
    today: Callable[[], date] = date.today,
):
    # Built once, not per-invocation.
    tool_specs = [
        ToolSpec(
            name=document_search_tool.name,
            description=document_search_tool.description,
            parameters=DocumentSearchInput.model_json_schema(),
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
        ToolSpec(
            name=escalate_ticket_tool.name,
            description=escalate_ticket_tool.description,
            parameters=EscalateTicketInput.model_json_schema(),
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

        if decision.tool_name == document_search_tool.name:
            try:
                args = DocumentSearchInput.model_validate(decision.arguments)
            except ValidationError:
                return {"decision": "respond", "planner_content": _CLARIFY_MESSAGE}
            return {"decision": "call_document_search_tool", "query": args.query}

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

        if decision.tool_name == escalate_ticket_tool.name:
            try:
                args = EscalateTicketInput.model_validate(decision.arguments)
            except ValidationError:
                return {"decision": "respond", "planner_content": _CLARIFY_MESSAGE}
            return {
                "decision": "call_escalate_ticket_tool",
                "ticket_id": args.ticket_id,
                "priority": args.priority,
                "reason": args.reason,
            }

        # decision.tool_name is either None (model chose to just respond)
        # or names a tool that doesn't exist (hallucinated) - both are
        # handled the same way: nothing to execute, fall back to whatever
        # plain-text reply the model gave.
        return {"decision": "respond", "planner_content": decision.content}

    def _call_document_search_tool(state: AgentState) -> dict[str, DocumentSearchResult]:
        # `_planner` only routes here when it has also set `query`.
        result = document_search_tool.run(DocumentSearchInput(query=state["query"]))
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

    def _call_escalate_ticket_tool(state: AgentState) -> dict[str, EscalateTicketResult]:
        # `_planner` only routes here when it has also set ticket_id,
        # priority, and reason. requester_id is never planner-set - it's
        # only ever present because the API layer put it there from the
        # verified JWT before graph.invoke() was called at all.
        result = escalate_ticket_tool.run(
            EscalateTicketInput(
                ticket_id=state["ticket_id"], priority=state["priority"], reason=state["reason"]
            ),
            requester_id=state["requester_id"],
        )
        # Notifying is this layer's job, not the tool's - the tool stays a
        # pure DB write with nothing to fake/mock in its own tests, and
        # this is the same call shape the decide endpoint will use later
        # for the employee confirmation email. get_manager_emails is only
        # called here, not for every request, the same reason `today` is
        # injected as a callable rather than eagerly evaluated - the query
        # only needs to run on the one path that actually needs it.
        if result.found:
            for manager_email in get_manager_emails():
                email_service.send(
                    to=manager_email,
                    subject=f"Approval needed: escalate ticket #{result.ticket_id}",
                    body=(
                        f"{state['reason']}\n\n"
                        f"Requested priority: {result.requested_priority.value}\n"
                        f"Approval request #{result.approval_request_id} is pending your review."
                    ),
                )
        return {"tool_result": result}

    def _respond(state: AgentState) -> dict[str, str]:
        if state.get("blocked"):
            return {"answer": "I can't help with that request."}

        result = state.get("tool_result")
        if result is None:
            planner_content = state.get("planner_content")
            if planner_content:
                return {"answer": planner_content}
            return {"answer": "I don't have a way to answer that yet."}

        if isinstance(result, DocumentSearchResult):
            if not result.found:
                return {"answer": "I couldn't find anything relevant in our documents."}
            synthesized = gateway.answer_from_context(
                state["question"], _format_context(result.chunks)
            )
            if synthesized.blocked:
                return {"answer": "I can't help with that request."}
            return {"answer": f"{synthesized.text}\n\n(Source: {result.chunks[0].source})"}

        if isinstance(result, TicketCountResult):
            return {"answer": f"{result.count} ticket(s) were opened in that period."}

        if isinstance(result, UnresolvedTicketsResult):
            if not result.tickets:
                return {"answer": "There are no unresolved tickets."}
            # One ticket per line, not semicolon-joined - the chat UI
            # preserves newlines (white-space: pre-wrap), so this renders
            # as a real list rather than a run-on sentence. Includes the
            # ticket id (needed to actually reference one for escalation)
            # and priority (previously invisible without opening a ticket).
            lines = "\n".join(
                f"#{ticket.id} [{ticket.priority.value}] {ticket.subject}"
                for ticket in result.tickets
            )
            return {"answer": f"{len(result.tickets)} unresolved ticket(s):\n{lines}"}

        if isinstance(result, EscalateTicketResult):
            if not result.found:
                return {"answer": f"I couldn't find ticket #{state['ticket_id']}."}
            return {
                "answer": (
                    f"I've proposed escalating ticket #{result.ticket_id} to "
                    f"{result.requested_priority.value} priority. This requires manager "
                    f"approval before it takes effect "
                    f"(request #{result.approval_request_id})."
                )
            }

        # Not a defensive no-op: unlike the state-field guarantees above
        # (which our own control flow makes provably true), "every
        # ToolResult member is handled" is a property that rots the moment a
        # new tool is added and this function isn't updated to match. Fail
        # loudly at the gap instead of a KeyError three layers away when
        # `answer` turns out unset.
        raise AssertionError(f"unhandled tool result type: {type(result)}")

    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("call_document_search_tool", _call_document_search_tool)
    graph.add_node("call_ticket_count_tool", _call_ticket_count_tool)
    graph.add_node("call_unresolved_tickets_tool", _call_unresolved_tickets_tool)
    graph.add_node("call_escalate_ticket_tool", _call_escalate_ticket_tool)
    graph.add_node("respond", _respond)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "call_document_search_tool": "call_document_search_tool",
            "call_ticket_count_tool": "call_ticket_count_tool",
            "call_unresolved_tickets_tool": "call_unresolved_tickets_tool",
            "call_escalate_ticket_tool": "call_escalate_ticket_tool",
            "respond": "respond",
        },
    )
    graph.add_edge("call_document_search_tool", "respond")
    graph.add_edge("call_ticket_count_tool", "respond")
    graph.add_edge("call_unresolved_tickets_tool", "respond")
    graph.add_edge("call_escalate_ticket_tool", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
