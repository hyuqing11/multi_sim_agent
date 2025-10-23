"""Streamlit dashboard for interacting with the FastAPI backend.
Run with: `streamlit run frontend/streamlit_app.py`
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import requests
import streamlit as st

API_BASE_URL = st.secrets.get("api_base_url", os.getenv("API_BASE_URL", "http://localhost:8000"))

st.set_page_config(page_title="DFT Copilot Control Panel", layout="wide")
st.title("DFT Copilot Control Panel")


def _format_timestamp(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=10, **kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network errors shown to user
        detail = getattr(exc.response, "text", "") if hasattr(exc, "response") else ""
        st.error(f"{method.upper()} {url} failed: {exc}\n{detail}")
        raise
    return response


def fetch_tasks() -> List[Dict[str, Any]]:
    response = _request("get", "/api/tasks")
    return response.json()


def create_task(prompt: str, goal: str | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"prompt": prompt}
    if goal:
        payload["goal"] = goal
    response = _request("post", "/api/tasks", json=payload)
    return response.json()


def approve_plan(task_id: str, accepted: bool, notes: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"accepted": accepted}
    if notes:
        payload["notes"] = notes
    response = _request("post", f"/api/tasks/{task_id}/plan/approval", json=payload)
    return response.json()


def execute_task(task_id: str) -> Dict[str, Any]:
    response = _request("post", f"/api/tasks/{task_id}/execute")
    return response.json()


def add_comment(task_id: str, step_id: str, body: str, author: str) -> Dict[str, Any]:
    payload = {"stepId": step_id, "body": body, "author": author}
    response = _request("post", f"/api/tasks/{task_id}/plan/comment", json=payload)
    return response.json()


def refresh_task(task_id: str) -> Dict[str, Any]:
    response = _request("get", f"/api/tasks/{task_id}")
    return response.json()


with st.sidebar:
    st.header("Create a new task")
    with st.form("create_task_form", clear_on_submit=True):
        prompt = st.text_area("Prompt", placeholder="Describe the structure workflow to run", height=120)
        goal = st.text_input("Goal (optional)")
        submitted = st.form_submit_button("Create task")
        if submitted:
            if not prompt.strip():
                st.warning("Please enter a prompt before creating a task.")
            else:
                try:
                    task = create_task(prompt.strip(), goal.strip() or None)
                except requests.RequestException:
                    st.stop()
                else:
                    st.success(f"Task created: {task['id']}")
                    st.session_state.setdefault("latest_task_id", task["id"])
                    st.rerun()

st.caption(f"Backend API: {API_BASE_URL}")

refresh_interval_ms = st.sidebar.slider("Auto-refresh interval (ms)", 0, 60000, 5000, step=1000)
# Auto-refresh implementation using sleep and rerun
if refresh_interval_ms:
    # Store last refresh time in session state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    time_since_refresh = time.time() - st.session_state.last_refresh
    if time_since_refresh * 1000 >= refresh_interval_ms:
        st.session_state.last_refresh = time.time()
        st.rerun()

try:
    tasks = fetch_tasks()
except requests.RequestException:
    st.stop()

if not tasks:
    st.info("No tasks created yet. Use the form on the left to start a workflow.")
else:
    st.subheader("Active tasks")
    for task in tasks:
        header = f"{task['prompt']} (status: {task['status']})"
        expanded = st.session_state.get("latest_task_id") == task["id"]
        with st.expander(header, expanded=expanded):
            meta_cols = st.columns(3)
            meta_cols[0].metric("Status", task["status"].title())
            meta_cols[1].metric("Created", _format_timestamp(task.get("createdAt")))
            meta_cols[2].metric("Updated", _format_timestamp(task.get("updatedAt")))

            if task.get("goal"):
                st.markdown(f"**Goal:** {task['goal']}")

            summary = task.get("summaryMarkdown")
            if summary:
                st.markdown("### Summary")
                st.markdown(summary)

            if task.get("toolEvents"):
                st.markdown("### Tool events")
                st.json(task["toolEvents"])

            plan_steps: Iterable[Dict[str, Any]] = task.get("plan", [])
            if plan_steps:
                st.markdown("### Plan")
                for step in plan_steps:
                    step_header = f"{step['title']} ({step['status']})"
                    with st.container():
                        st.markdown(f"**{step_header}**")
                        if step.get("description"):
                            st.write(step["description"])
                        if step.get("assignedAgent"):
                            st.caption(f"Assigned to: {step['assignedAgent']}")

                        comments: List[Dict[str, Any]] = step.get("comments", [])
                        if comments:
                            st.markdown("Comments:")
                            for comment in comments:
                                timestamp = _format_timestamp(comment.get("createdAt"))
                                st.write(f"- {comment['author']} ({timestamp}): {comment['body']}")

                        with st.form(key=f"comment_form_{task['id']}_{step['id']}", clear_on_submit=True):
                            comment_body = st.text_area("Add a comment", key=f"comment_{task['id']}_{step['id']}")
                            author = st.text_input("Author", value="User", key=f"author_{task['id']}_{step['id']}")
                            submit_comment = st.form_submit_button("Submit comment")
                            if submit_comment:
                                if not comment_body.strip():
                                    st.warning("Comment cannot be empty.")
                                else:
                                    try:
                                        add_comment(task["id"], step["id"], comment_body.strip(), author.strip() or "User")
                                    except requests.RequestException:
                                        st.stop()
                                    else:
                                        st.success("Comment added.")
                                        st.rerun()

            action_cols = st.columns(4)
            if action_cols[0].button("Refresh", key=f"refresh_{task['id']}"):
                try:
                    refreshed = refresh_task(task["id"])
                except requests.RequestException:
                    st.stop()
                else:
                    st.session_state["latest_task_id"] = refreshed["id"]
                    st.rerun()

            with action_cols[1]:
                approval_notes = st.text_input("Approval notes", key=f"notes_{task['id']}", placeholder="Optional notes")
            if action_cols[2].button("Approve plan", key=f"approve_{task['id']}"):
                try:
                    approve_plan(task["id"], True, st.session_state.get(f"notes_{task['id']}") or None)
                except requests.RequestException:
                    st.stop()
                else:
                    st.success("Plan approved.")
                    st.rerun()
            if action_cols[3].button("Request changes", key=f"reject_{task['id']}"):
                try:
                    approve_plan(task["id"], False, st.session_state.get(f"notes_{task['id']}") or None)
                except requests.RequestException:
                    st.stop()
                else:
                    st.info("Plan sent back for changes.")
                    st.rerun()

            if st.button("Execute task", key=f"execute_{task['id']}"):
                try:
                    execute_task(task["id"])
                except requests.RequestException:
                    st.stop()
                else:
                    st.success("Execution started.")
                    st.rerun()

            st.divider()

st.markdown(
    """
    <small>Use the sidebar to create new tasks. The dashboard polls the FastAPI backend periodically to keep the
    task list up to date. Actions such as approving a plan or executing a task immediately trigger API calls against
    the backend.</small>
    """,
    unsafe_allow_html=True,
)