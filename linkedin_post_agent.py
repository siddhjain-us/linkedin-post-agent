"""
Workflow 1: Stateful LinkedIn High-Signal Post Generator (LangGraph + OpenRouter + Composio + SQLite)

Demonstrates:
- Shared TypedDict state with bounded revision tracking
- Unconditional edges (START -> ingest_update -> draft_post -> quality_eval)
- Conditional edges (quality_eval loop + human_review approval/edit router)
- Durable Human-in-the-Loop (HITL) via interrupt() and Command(resume=...)
- Fault-tolerant crash recovery via graph.invoke(None, config)
- Post-HITL deterministic LinkedIn publishing via Composio (never autonomous LLM posting)
- Persistent SQLite checkpointer (SqliteSaver) that survives script restarts
- StateSnapshot inspection via graph.get_state_history(config)

Setup:
    python3 -m venv .venv && source .venv/bin/activate
    pip install -U langgraph langgraph-checkpoint-sqlite langchain-openai composio pydantic python-dotenv
    export OPENROUTER_API_KEY="sk-or-v1-..."
    export OPENROUTER_MODEL="openrouter/free"
    export COMPOSIO_API_KEY="ak_..."
    export COMPOSIO_USER_ID="siddh"
    export LINKEDIN_DRY_RUN="true"   # Set to "false" once ready to publish live

First-time LinkedIn OAuth (once):
    python linkedin_post_agent.py --connect

Run:
    python linkedin_post_agent.py
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from composio import Composio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Model Client (OpenRouter now; 1-line swap to local Mac Mini MLX / Ollama)
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.4) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Set it with: export OPENROUTER_API_KEY='sk-or-...'"
        )
    primary_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    fallback_models = [primary_model]
    if primary_model != "openrouter/free":
        fallback_models.append("openrouter/free")

    # To run locally on a Mac Mini with Apple MLX or Ollama, swap base_url to:
    # "http://localhost:8080/v1" (mlx_lm.server) or "http://localhost:11434/v1" (Ollama)
    return ChatOpenAI(
        model=primary_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_retries=3,
        extra_body={"models": fallback_models},
    )


# ---------------------------------------------------------------------------
# 2. Graph State Schema & Structured Critic Output
# ---------------------------------------------------------------------------
class PostState(TypedDict):
    raw_update: str          # What you built or learned
    draft: str               # Latest LinkedIn post draft
    critique: str            # Automated critic feedback
    score: int               # 1-10 signal quality score
    revision_count: int      # Guardrail to prevent infinite loops
    human_feedback: str      # Optional edit note from HITL interrupt()
    approved_path: str       # Local Markdown archive path
    publish_status: str      # Dry-run or Composio LinkedIn API result


class EvalRubric(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Signal quality score from 1 to 10.")
    critique: str = Field(..., description="Specific, actionable critique if score < 8.")


# ---------------------------------------------------------------------------
# 3. Graph Nodes
# ---------------------------------------------------------------------------
def ingest_update(state: PostState) -> dict:
    """Normalize input and initialize loop counters."""
    return {
        "raw_update": state["raw_update"].strip(),
        "revision_count": state.get("revision_count", 0),
        "human_feedback": state.get("human_feedback", ""),
    }


def draft_post(state: PostState) -> dict:
    """Stateful writer node: drafts or revises a high-signal technical LinkedIn post."""
    llm = get_llm(temperature=0.5)
    feedback_block = ""
    if state.get("human_feedback"):
        feedback_block = f"\nHuman reviewer feedback (highest priority):\n{state['human_feedback']}\n"
    elif state.get("critique"):
        feedback_block = f"\nPrevious critic feedback:\n{state['critique']}\n"

    prompt = (
        "Write a concise, high-signal technical LinkedIn post (120-180 words) based on this build update.\n"
        "Rules: concrete technical hook in line 1, zero corporate buzzwords, no hashtag spam (max 2), "
        "explain one real engineering tradeoff you learned. Output only the post text.\n\n"
        f"Build update:\n{state['raw_update']}\n"
        f"{feedback_block}"
    )
    response = llm.invoke(prompt)
    return {
        "draft": response.content.strip(),
        "revision_count": state.get("revision_count", 0) + 1,
        "human_feedback": "",
    }


def quality_eval(state: PostState) -> dict:
    """Stateless evaluator node: scores the draft against a strict technical rubric."""
    llm = get_llm(temperature=0.0)
    eval_prompt = (
        "Score this LinkedIn post from 1-10 for technical signal, specificity, and readability. "
        "Deduct points for generic AI hype, fluff, or missing concrete details.\n\n"
        f"Draft:\n{state['draft']}"
    )
    try:
        result: EvalRubric = llm.with_structured_output(EvalRubric).invoke(eval_prompt)
        return {"score": result.score, "critique": result.critique}
    except Exception:
        # Fallback for :free models that do not support native tool-calling schemas
        raw = llm.invoke(
            eval_prompt
            + '\n\nRespond ONLY with valid JSON: {"score": <int 1-10>, "critique": "<1 sentence>"}'
        ).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            return {
                "score": max(1, min(10, int(parsed.get("score", 8)))),
                "critique": str(parsed.get("critique", "Looks solid.")),
            }
        return {"score": 8, "critique": "Auto-critic fallback: passed to human review."}


def human_review(state: PostState) -> Command[Literal["draft_post", "publish_post"]]:
    """
    Durable Human-in-the-Loop node.
    Calling interrupt() freezes graph state into SQLite and returns control to the caller.
    You can exit the script, reboot your laptop, and resume days later with the same thread_id.
    """
    decision = interrupt(
        {
            "draft": state["draft"],
            "score": state["score"],
            "critique": state["critique"],
            "instructions": "Reply with {'action': 'approve'} or {'action': 'edit', 'note': '...'}",
        }
    )

    if decision.get("action") == "approve":
        return Command(goto="publish_post")

    return Command(
        goto="draft_post",
        update={"human_feedback": decision.get("note", "Tighten the hook and add specifics.")},
    )


def _resolve_linkedin_author_urn(composio: Composio, user_id: str) -> str:
    """Resolve urn:li:person:<id> from env or Composio's LINKEDIN_GET_MY_INFO tool."""
    if os.getenv("LINKEDIN_AUTHOR_URN"):
        return os.environ["LINKEDIN_AUTHOR_URN"]

    info = composio.tools.execute(
        slug="LINKEDIN_GET_MY_INFO",
        user_id=user_id,
        arguments={},
        dangerously_skip_version_check=True,
    )
    data = info.get("data", {}) if isinstance(info, dict) else {}
    sub = data.get("sub") or data.get("response_data", {}).get("sub") or data.get("id")
    if not sub:
        raise RuntimeError(
            f"Could not resolve LinkedIn author URN from LINKEDIN_GET_MY_INFO response: {info}. "
            "Set LINKEDIN_AUTHOR_URN='urn:li:person:<your_id>' or connect LinkedIn first."
        )
    return sub if str(sub).startswith("urn:li:") else f"urn:li:person:{sub}"


def publish_post(state: PostState) -> dict:
    """
    Deterministic post-HITL publisher node.
    Always archives the approved post locally, then publishes to LinkedIn via Composio
    when LINKEDIN_DRY_RUN is 'false'.
    """
    out_dir = Path("approved_posts")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"linkedin_post_{ts}.md"
    out_path.write_text(state["draft"] + "\n", encoding="utf-8")

    dry_run = os.getenv("LINKEDIN_DRY_RUN", "true").lower() != "false"
    if dry_run:
        return {
            "approved_path": str(out_path),
            "publish_status": "DRY_RUN (saved locally; set LINKEDIN_DRY_RUN=false to post live via Composio)",
        }

    if not os.getenv("COMPOSIO_API_KEY"):
        raise RuntimeError("Missing COMPOSIO_API_KEY. Set it before setting LINKEDIN_DRY_RUN=false.")

    user_id = os.getenv("COMPOSIO_USER_ID", "siddh")
    composio = Composio()
    author_urn = _resolve_linkedin_author_urn(composio, user_id)

    result = composio.tools.execute(
        slug="LINKEDIN_CREATE_LINKED_IN_POST",
        user_id=user_id,
        arguments={
            "author": author_urn,
            "commentary": state["draft"],
            "visibility": "PUBLIC",
        },
        dangerously_skip_version_check=True,
    )

    if isinstance(result, dict) and not result.get("successful", True):
        return {
            "approved_path": str(out_path),
            "publish_status": f"ERROR: {result.get('error', result)}",
        }

    return {
        "approved_path": str(out_path),
        "publish_status": f"PUBLISHED via Composio ({result})",
    }


# ---------------------------------------------------------------------------
# 4. Conditional Edge Router & Graph Compilation
# ---------------------------------------------------------------------------
def route_after_eval(state: PostState) -> Literal["draft_post", "human_review"]:
    """Loop back to draft_post if score < 8 and we haven't hit max revisions."""
    if state["score"] < 8 and state["revision_count"] < 2:
        return "draft_post"
    return "human_review"


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(PostState)

    builder.add_node("ingest_update", ingest_update)
    builder.add_node("draft_post", draft_post)
    builder.add_node("quality_eval", quality_eval)
    builder.add_node("human_review", human_review)
    builder.add_node("publish_post", publish_post)

    # Unconditional edges
    builder.add_edge(START, "ingest_update")
    builder.add_edge("ingest_update", "draft_post")
    builder.add_edge("draft_post", "quality_eval")
    builder.add_edge("publish_post", END)

    # Conditional edge (Evaluator-Optimizer loop)
    builder.add_conditional_edges(
        "quality_eval",
        route_after_eval,
        {"draft_post": "draft_post", "human_review": "human_review"},
    )

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 5. One-Time LinkedIn Connection Helper & Interactive CLI
# ---------------------------------------------------------------------------
def connect_linkedin_account() -> None:
    """Generates a Composio OAuth link to connect your LinkedIn account once."""
    if not os.getenv("COMPOSIO_API_KEY"):
        raise RuntimeError("Set COMPOSIO_API_KEY first.")
    user_id = os.getenv("COMPOSIO_USER_ID", "siddh")
    composio = Composio()
    session = composio.create(user_id=user_id)
    connection_request = session.authorize("linkedin")
    print("\nOpen this URL in your browser to connect LinkedIn via Composio:")
    print(connection_request.redirect_url)
    print("\nWaiting for authorization to complete...")
    connection_request.wait_for_connection()
    print("LinkedIn connected! You can now run: LINKEDIN_DRY_RUN=false python linkedin_post_agent.py")


def print_snapshots(graph, config: dict) -> None:
    print("\n--- SQLite StateSnapshot History (newest first) ---")
    for snap in graph.get_state_history(config):
        cid = snap.config["configurable"].get("checkpoint_id", "")[:12]
        score = snap.values.get("score", "-")
        rev = snap.values.get("revision_count", 0)
        print(f"checkpoint={cid} | next={snap.next} | rev={rev} | score={score}")


def main() -> None:
    if "--connect" in sys.argv:
        connect_linkedin_account()
        return

    conn = sqlite3.connect("agent_state.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(checkpointer)

    thread_id = input("Thread ID (press Enter for 'post-001', or reuse a paused ID): ").strip() or "post-001"
    config = {"configurable": {"thread_id": thread_id}}

    # Distinguish between (a) paused at interrupt(), (b) crashed mid-run, and (c) brand new thread
    current_state = graph.get_state(config)
    pending_interrupts = [
        intr for task in (current_state.tasks or ()) for intr in (task.interrupts or ())
    ]

    if pending_interrupts:
        print(f"\nResuming HITL interrupt on thread '{thread_id}' at node(s): {current_state.next}")
        result = {"__interrupt__": pending_interrupts}
    elif current_state.next:
        # Fault-tolerant recovery: retries unfinished node (e.g. after a 429) without re-running prior steps
        print(f"\nRecovering unfinished thread '{thread_id}' from node(s): {current_state.next}")
        result = graph.invoke(None, config=config)
    else:
        raw_update = input("\nWhat did you build or learn? (1-3 sentences):\n> ").strip()
        result = graph.invoke({"raw_update": raw_update}, config=config)

    # HITL Loop: pauses cleanly whenever human_review calls interrupt()
    while "__interrupt__" in result and result["__interrupt__"]:
        payload = result["__interrupt__"][0].value
        print("\n================= DRAFT FOR HUMAN REVIEW =================")
        print(payload["draft"])
        print("==========================================================")
        print(f"Auto-Critic Score: {payload['score']}/10 | Note: {payload['critique']}")

        choice = input("\nType [a]pprove, [e]dit, or [q]uit (state stays saved in SQLite): ").strip().lower()
        if choice == "q":
            print(f"\nSaved! Re-run this script anytime with thread_id='{thread_id}' to resume.")
            print_snapshots(graph, config)
            return
        if choice == "a":
            result = graph.invoke(Command(resume={"action": "approve"}), config=config)
        else:
            note = input("What should the agent change?\n> ").strip()
            result = graph.invoke(Command(resume={"action": "edit", "note": note}), config=config)

    print(f"\nLocal archive: {result.get('approved_path')}")
    print(f"Publish status: {result.get('publish_status')}")
    print_snapshots(graph, config)
    print("\n--- Mermaid Graph (paste into mermaid.live for your call) ---")
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
