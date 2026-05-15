"""Exercise 3 - Escalation branch with reviewer Q&A."""

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
                {
                    "role": "system",
                    "content": STRICT_REVIEW_SYSTEM_PROMPT,
                },
                {"role": "user", "content": f"Title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}"},
            ]
        )
    console.print(
        f"  [green]OK[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.escalation_questions)} question(s)"
    )
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
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def node_escalate(state: ReviewState) -> dict:
    analysis = state["analysis"]
    questions = analysis.escalation_questions or [
        "What is the intended behavior of this PR?",
        "Are there security, migration, or compatibility constraints the review should consider?",
    ]
    response = interrupt(
        {
            "kind": "escalation",
            "pr_url": state["pr_url"],
            "confidence": analysis.confidence,
            "confidence_reasoning": analysis.confidence_reasoning,
            "summary": analysis.summary,
            "risk_factors": analysis.risk_factors,
            "questions": questions,
        }
    )
    return {"escalation_answers": response}


def node_synthesize(state: ReviewState) -> dict:
    console.print("[cyan]-> synthesize[/cyan]")
    answers = state.get("escalation_answers") or {}
    qa = "\n".join(f"Q: {question}\nA: {answer}" for question, answer in answers.items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "Use the reviewer answers to produce a refined PRAnalysis.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Original diff:\n{state['pr_diff']}\n\n"
                        f"Initial summary: {state['analysis'].summary}\n\n"
                        f"Reviewer Q&A:\n{qa}"
                    ),
                },
            ]
        )
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def _render_comment_body(state: ReviewState) -> str:
    analysis = state["analysis"]
    lines = [f"### Automated review (confidence {analysis.confidence:.0%})", "", analysis.summary, ""]
    for comment in analysis.comments:
        lines.append(f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` - {comment.body}")
    if state.get("human_feedback"):
        lines.append(f"\nReviewer note: {state['human_feedback']}")
    if state.get("escalation_answers"):
        lines.append("\nReviewer answered escalation questions:")
        for question, answer in state["escalation_answers"].items():
            lines.append(f"> **{question}** {answer}")
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
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]skipping comment[/yellow] choice={state.get('human_choice')}")
    return {"final_action": "rejected"}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - no human needed[/dim]")
    return {}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("commit", node_commit)
    g.add_node("escalate", node_escalate)
    g.add_node("synthesize", node_synthesize)

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
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload: dict):
    if payload["kind"] == "approval_request":
        console.print(
            Panel.fit(
                payload["summary"],
                title=f"Approve? conf={payload['confidence']:.0%}",
                border_style="green",
            )
        )
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}

    if payload["kind"] == "escalation":
        console.print(
            Panel.fit(
                payload["summary"],
                title=f"Escalation conf={payload['confidence']:.0%}",
                border_style="yellow",
            )
        )
        return {question: console.input(f"Q: {question}\nA: ").strip() for question in payload["questions"]}

    raise ValueError(payload["kind"])


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 3 - escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = app.invoke(Command(resume=handle_interrupt(payload)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
