"""Exercise 1 - Confidence scoring + routing."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
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
                {
                    "role": "user",
                    "content": f"Title: {state['pr_title']}\n\nDiff:\n{state['pr_diff']}",
                },
            ]
        )
    console.print(
        f"  [green]OK[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s)"
    )
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    decision = route_decision_for_analysis(state["analysis"])
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold]")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]AUTO APPROVE[/green] - high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]HUMAN APPROVAL[/yellow] - exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]ESCALATE[/red] - exercise 3 will ask reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
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
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 - confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
