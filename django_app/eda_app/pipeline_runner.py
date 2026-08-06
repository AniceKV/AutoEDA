"""
pipeline_runner.py
------------------
Thread-safe background pipeline executor.
Stores per-session state in an in-memory dict so Django views can poll status.
Also maintains a per-session log buffer for live streaming to the frontend.
"""
import os
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional

# Add the AutoEDA root to sys.path so we can import the existing modules
AUTOEDA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if AUTOEDA_ROOT not in sys.path:
    sys.path.insert(0, AUTOEDA_ROOT)

# ─── In-memory session state store ────────────────────────────────────────────
# { session_id: { "status": "running"|"done"|"error"|"question",
#                 "message": str,
#                 "result": dict | None,
#                 "conversation_history": list,
#                 "question": str | None } }
_PIPELINE_STATE: Dict[str, Dict[str, Any]] = {}
_STATE_LOCK = threading.Lock()

# ─── In-memory log buffer ─────────────────────────────────────────────────────
_PIPELINE_LOGS: Dict[str, List[str]] = {}
_LOG_LOCK = threading.Lock()
_MAX_LOG_LINES = 500


def get_state(session_id: str) -> Optional[Dict[str, Any]]:
    with _STATE_LOCK:
        return _PIPELINE_STATE.get(session_id)


def set_state(session_id: str, state: Dict[str, Any]):
    with _STATE_LOCK:
        _PIPELINE_STATE[session_id] = state


def clear_state(session_id: str):
    with _STATE_LOCK:
        _PIPELINE_STATE.pop(session_id, None)
    clear_log(session_id)


def append_log(session_id: str, line: str):
    """Append a log line to the session buffer (thread-safe)."""
    with _LOG_LOCK:
        buf = _PIPELINE_LOGS.setdefault(session_id, [])
        buf.append(line)
        if len(buf) > _MAX_LOG_LINES:
            _PIPELINE_LOGS[session_id] = buf[-_MAX_LOG_LINES:]


def get_log(session_id: str) -> List[str]:
    """Return a copy of the current log buffer for a session."""
    with _LOG_LOCK:
        return list(_PIPELINE_LOGS.get(session_id, []))


def clear_log(session_id: str):
    with _LOG_LOCK:
        _PIPELINE_LOGS.pop(session_id, None)


# ─── Background runner ─────────────────────────────────────────────────────────

def _run(session_id: str, data_path: str, user_request: str,
         workspace_dir: str, generate_summary: bool,
         conversation_history: list):
    """
    Runs inside a daemon Thread. Imports agent_loop lazily to avoid
    importing at module load time (heavy deps).
    """
    try:
        from agent_loop import run_tool_based_eda

        clear_log(session_id)
        append_log(session_id, "▶ Pipeline initialised — profiling dataset…")

        set_state(session_id, {
            "status": "running",
            "message": "Profiling dataset and building analysis plan…",
            "result": None,
            "conversation_history": conversation_history,
            "question": None,
        })

        # Pass a status callback if agent_loop supports it
        import inspect
        sig = inspect.signature(run_tool_based_eda)
        kwargs: Dict[str, Any] = dict(
            data_path=data_path,
            user_request=user_request,
            workspace_dir=workspace_dir,
            generate_summary=generate_summary,
            conversation_history=conversation_history,
        )
        if "status_callback" in sig.parameters:
            def _cb(msg: str):
                append_log(session_id, msg)
                with _STATE_LOCK:
                    s = _PIPELINE_STATE.get(session_id)
                    if s and s.get("status") == "running":
                        s["message"] = msg
            kwargs["status_callback"] = _cb

        result = run_tool_based_eda(**kwargs)

        if result.get("status") == "question":
            append_log(session_id, "❓ Agent requires clarification from user.")
            set_state(session_id, {
                "status": "question",
                "message": "Agent requires clarification.",
                "result": result,
                "conversation_history": result.get("conversation_history", []),
                "question": result.get("question", ""),
            })
        else:
            append_log(session_id, "✅ Analysis complete — all tasks finished.")
            set_state(session_id, {
                "status": "done",
                "message": "Analysis complete.",
                "result": result,
                "conversation_history": result.get("conversation_history", []),
                "question": None,
            })

    except Exception as exc:
        tb = traceback.format_exc()
        err_msg = f"Pipeline error: {exc}"
        append_log(session_id, f"❌ {err_msg}")
        set_state(session_id, {
            "status": "error",
            "message": err_msg,
            "traceback": tb,
            "result": None,
            "conversation_history": conversation_history,
            "question": None,
        })


def launch_pipeline(session_id: str, data_path: str, user_request: str,
                    workspace_dir: str, generate_summary: bool,
                    conversation_history: list):
    """Starts the pipeline in a background daemon thread."""
    set_state(session_id, {
        "status": "running",
        "message": "Initialising pipeline…",
        "result": None,
        "conversation_history": conversation_history,
        "question": None,
    })
    t = threading.Thread(
        target=_run,
        args=(session_id, data_path, user_request, workspace_dir,
              generate_summary, conversation_history),
        daemon=True,
    )
    t.start()
