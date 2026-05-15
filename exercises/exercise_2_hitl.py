"""Exercise 2 - HITL with interrupt() + Command(resume=...)."""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.review_policy import STRICT_REVIEW_SYSTEM_PROMPT
from common.schemas import (
    PRAnalysis,
    ReviewState,
    route_decision_for_analysis,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis: PRAnalysis = llm.invoke(
            [
                {"role": "system", "content": STRICT_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}"},
            ]
        )
    console.print(f"  [green]OK[/green] confidence={analysis.confidence:.0%}")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    decision = route_decision_for_analysis(state["analysis"])
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold]")
    return {"decision": decision}


def node_human_approval(state: ReviewState) -> dict:
    analysis = state["analysis"]
    response = interrupt(
        {
            "kind": "approval_request",
            "pr_url": state["pr_url"],
            "confidence": analysis.confidence,
            "confidence_reasoning": analysis.confidence_reasoning,
            "summary": analysis.summary,
            "comments": [comment.model_dump() for comment in analysis.comments],
            "diff_preview": state["pr_diff"][:2000],
        }
    )
    return {
        "human_choice": response.get("choice"),
        "human_feedback": response.get("feedback"),
    }


def _render_comment_body(state: ReviewState) -> str:
    analysis = state["analysis"]
    lines = [f"### Automated review (confidence {analysis.confidence:.0%})", "", analysis.summary, ""]
    for comment in analysis.comments:
        lines.append(f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` - {comment.body}")
    if state.get("human_feedback"):
        lines.append(f"\nReviewer note: {state['human_feedback']}")
    return "\n".join(lines)


def _post(state: ReviewState, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as exc:
        console.print(f"  [red]post failed:[/red] {exc}")
        return "commit_failed"


def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    if state.get("decision") == "auto_approve":
        return {"final_action": _post(state, "auto_approved")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]skipping comment[/yellow] choice={state.get('human_choice')}")
    return {"final_action": "rejected"}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - no human needed[/dim]")
    return {}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]ESCALATE[/red] - exercise 3 implements escalation Q&A")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("commit", node_commit)
    g.add_node("escalate", node_escalate)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    g.add_edge("auto_approve", "commit")
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=MemorySaver())


def prompt_human(payload: dict) -> dict:
    console.print(
        Panel.fit(
            f"[bold]Confidence:[/bold] {payload['confidence']:.0%}\n"
            f"[dim]{payload['confidence_reasoning']}[/dim]\n\n"
            f"[bold]Summary:[/bold] {payload['summary']}",
            title="Approval request",
            border_style="green",
        )
    )
    for comment in payload.get("comments", []):
        console.print(
            f"  [{comment['severity']}] {comment['file']}:{comment.get('line') or '?'} - {comment['body']}"
        )
    if payload.get("diff_preview"):
        console.print("\n[dim]--- diff preview ---[/dim]")
        console.print(payload["diff_preview"])

    choice = ""
    while choice not in {"approve", "reject", "edit"}:
        choice = console.input("\n[bold]Choice (approve/reject/edit)?[/bold] ").strip().lower()
    feedback = console.input("Feedback: ").strip()
    return {"choice": choice, "feedback": feedback}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 2 - HITL with interrupt()[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        answer = prompt_human(payload)
        result = app.invoke(Command(resume=answer), cfg)

    console.rule("Done")
    console.print(result.get("final_action"))


if __name__ == "__main__":
    main()
