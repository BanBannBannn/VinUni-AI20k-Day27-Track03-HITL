"""Exercise 5 - Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


st.set_page_config(page_title="HITL PR Review", layout="wide")


def init_state() -> None:
    defaults = {
        "thread_id": None,
        "pr_url": "",
        "interrupt_payload": None,
        "final": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def recent_sessions() -> list[dict]:
    path = Path(db_path())
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MIN(timestamp) AS started,
                       MAX(timestamp) AS last_event,
                       MAX(risk_level) AS worst_risk,
                       COUNT(*) AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 10
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as checkpointer:
        await checkpointer.setup()
        app = build_graph(checkpointer)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


def render_approval_card(payload: dict) -> dict | None:
    st.subheader(f"Approval requested - confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** "
            f"`{comment['file']}:{comment.get('line') or '?'}` - {comment['body']}"
        )

    with st.expander("Diff preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback", key="approval_feedback")
    approve, reject, edit = st.columns(3)
    if approve.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if reject.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if edit.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation - confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers = {
            question: st.text_input(question, key=f"q_{index}")
            for index, question in enumerate(payload.get("questions", []))
        }
        if st.form_submit_button("Submit answers"):
            return answers
    return None


init_state()
st.title("HITL PR Review Agent")

with st.sidebar:
    st.header("Session")
    if st.session_state.thread_id:
        st.code(st.session_state.thread_id)
    st.header("Recent sessions")
    sessions = recent_sessions()
    if not sessions:
        st.caption("No audit sessions yet.")
    for index, session in enumerate(sessions):
        label = f"{session['worst_risk']} | {session['events']} events"
        if st.button(label, key=f"session_{index}", use_container_width=True):
            st.session_state.thread_id = session["thread_id"]
            st.session_state.pr_url = session["pr_url"]
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.rerun()
        st.caption(session["pr_url"])

with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")

if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR and asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

payload = st.session_state.interrupt_payload
if payload is not None:
    if payload["kind"] == "approval_request":
        answer = render_approval_card(payload)
    else:
        answer = render_escalation_card(payload)

    if answer is not None:
        with st.spinner("Resuming graph..."):
            result = asyncio.run(
                run_graph(st.session_state.pr_url, st.session_state.thread_id, resume_value=answer)
            )
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"{action} - comment posted to {st.session_state.pr_url}")
        st.link_button("View PR on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected - no comment posted")
    else:
        st.info(f"final_action = {action}")

    st.caption(
        f"thread_id = {st.session_state.thread_id} | "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
