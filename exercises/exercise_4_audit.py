"""Exercise 4 - Structured SQLite audit trail + durable checkpointer."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.review_policy import STRICT_REVIEW_SYSTEM_PROMPT
from common.schemas import (
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
    route_decision_for_analysis,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state: ReviewState, entry: AuditEntry) -> None:
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


async def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    started = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="fetch_pr",
            confidence=0.0,
            risk_level="med",
            decision="pending",
            reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    started = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis: PRAnalysis = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": STRICT_REVIEW_SYSTEM_PROMPT,
                },
                {"role": "user", "content": f"Title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}"},
            ]
        )
    console.print(f"  [green]OK[/green] confidence={analysis.confidence:.0%}")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="analyze",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="pending",
            reason=analysis.confidence_reasoning,
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {"analysis": analysis}


async def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    started = time.monotonic()
    confidence = state["analysis"].confidence
    decision = route_decision_for_analysis(state["analysis"])
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold]")
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="route",
            confidence=confidence,
            risk_level=risk_level_for(confidence),
            decision=decision,
            reason="Routed by confidence threshold",
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {"decision": decision}


async def node_human_approval(state: ReviewState) -> dict:
    started = time.monotonic()
    analysis = state["analysis"]
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="pending",
            reason="Waiting for human approval",
            execution_time_ms=0,
        ),
    )
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
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            reviewer_id=os.environ.get("GITHUB_USER"),
            decision=response.get("choice", "pending"),
            reason=response.get("feedback"),
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


async def node_escalate(state: ReviewState) -> dict:
    started = time.monotonic()
    analysis = state["analysis"]
    questions = analysis.escalation_questions or [
        "What is the intended behavior of this PR?",
        "Are there security, migration, or compatibility constraints the review should consider?",
    ]
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="escalate",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            decision="escalate",
            reason="Low confidence, waiting for reviewer answers",
            execution_time_ms=0,
        ),
    )
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
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="escalate",
            confidence=analysis.confidence,
            risk_level=risk_level_for(analysis.confidence),
            reviewer_id=os.environ.get("GITHUB_USER"),
            decision="answered",
            reason="Reviewer answered escalation questions",
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {"escalation_answers": response}


async def node_synthesize(state: ReviewState) -> dict:
    console.print("[cyan]-> synthesize[/cyan]")
    started = time.monotonic()
    answers = state.get("escalation_answers") or {}
    qa = "\n".join(f"Q: {question}\nA: {answer}" for question, answer in answers.items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke(
            [
                {"role": "system", "content": "Use reviewer answers to produce a refined PRAnalysis."},
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
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="synthesize",
            confidence=refined.confidence,
            risk_level=risk_level_for(refined.confidence),
            decision="synthesized",
            reason=f"Refined review from {state['analysis'].confidence:.0%} to {refined.confidence:.0%}",
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
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


def _post(state: ReviewState) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
        return "committed"
    except Exception as exc:
        console.print(f"  [red]post failed:[/red] {exc}")
        return "commit_failed"


async def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    started = time.monotonic()
    if state.get("decision") == "auto_approve":
        action = _post(state)
        final_action = "auto_approved" if action == "committed" else action
    elif state.get("escalation_answers"):
        action = _post(state)
        final_action = "committed_after_escalation" if action == "committed" else action
    elif state.get("human_choice") == "approve":
        action = _post(state)
        final_action = action
    else:
        console.print(f"  [yellow]skipping comment[/yellow] choice={state.get('human_choice')}")
        final_action = "rejected"
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="commit",
            confidence=state["analysis"].confidence,
            risk_level=risk_level_for(state["analysis"].confidence),
            reviewer_id=os.environ.get("GITHUB_USER") if state.get("human_choice") else None,
            decision=final_action,
            reason=f"Final action: {final_action}",
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {"final_action": final_action}


async def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - no human needed[/dim]")
    started = time.monotonic()
    confidence = state["analysis"].confidence
    await audit(
        state,
        AuditEntry(
            agent_id=AGENT_ID,
            action="auto_approve",
            confidence=confidence,
            risk_level=risk_level_for(confidence),
            decision="auto",
            reason=f"High confidence ({confidence:.0%}); continuing to commit without human review",
            execution_time_ms=int((time.monotonic() - started) * 1000),
        ),
    )
    return {}


def build_graph(checkpointer):
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
    return g.compile(checkpointer=checkpointer)


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


async def run(pr_url: str, thread_id: str | None) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--thread", help="Resume an existing thread")
    args = parser.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
