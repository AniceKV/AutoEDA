"""
views.py — All Django views for AutoEDA Pro.
"""
import glob
import json
import os
import re
import sys
import uuid

import pandas as pd
from django.conf import settings
from django.http import (FileResponse, HttpResponse, JsonResponse,
                         StreamingHttpResponse)
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.clickjacking import xframe_options_sameorigin

# Add AutoEDA root to sys.path
AUTOEDA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOEDA_ROOT = os.path.dirname(AUTOEDA_ROOT)  # go up one more from django_app/
if AUTOEDA_ROOT not in sys.path:
    sys.path.insert(0, AUTOEDA_ROOT)

from profiler import calculate_column_stats
from . import pipeline_runner

# ─── Numeric type helper ───────────────────────────────────────────────────────

def _safe_val(v):
    """Convert numpy/pandas scalars to JSON-safe Python types."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except ImportError:
        pass
    return v


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _session_id(request) -> str:
    """Get or create a unique session ID."""
    if "pipeline_session_id" not in request.session:
        request.session["pipeline_session_id"] = str(uuid.uuid4())
    return request.session["pipeline_session_id"]


def _workspace_dir(session_id: str) -> str:
    """Return per-session sandbox directory."""
    path = os.path.join(AUTOEDA_ROOT, "sandbox_run", session_id)
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_eda_dir(session_id: str, dataset_name: str) -> str | None:
    """Return the EDA output directory for the given dataset, or None."""
    candidate = os.path.join(AUTOEDA_ROOT, "EDA", dataset_name)
    if os.path.exists(candidate):
        return candidate
    ws = _workspace_dir(session_id)
    if os.path.exists(ws):
        return ws
    return None


def _dataset_overview(csv_path: str) -> dict:
    """Return quick overview stats for the dataset metrics bar."""
    try:
        df = pd.read_csv(csv_path)
        num_rows, num_cols = df.shape
        num_numeric = sum(
            pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            for c in df.columns
        )
        num_cat = num_cols - num_numeric
        total_cells = num_rows * num_cols
        missing_cells = int(df.isnull().sum().sum())
        missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
        return {
            "rows": f"{num_rows:,}",
            "cols": num_cols,
            "numeric": num_numeric,
            "categorical": num_cat,
            "missing_pct": missing_pct,
        }
    except Exception:
        return {}


def _column_profiles(csv_path: str) -> list:
    """Return list of column profile dicts for the Variable Profiles tab."""
    try:
        df = pd.read_csv(csv_path)
        stats_list = calculate_column_stats(df)
        result = []
        for s in stats_list:
            col_name = s["column"]
            is_num = (
                pd.api.types.is_numeric_dtype(df[col_name])
                and not pd.api.types.is_bool_dtype(df[col_name])
            )
            top_vals = None
            if not is_num:
                tv = df[col_name].value_counts().head(5).reset_index()
                tv.columns = ["Value", "Count"]
                tv["Pct"] = (tv["Count"] / len(df) * 100).round(1).astype(str) + "%"
                top_vals = tv.to_dict("records")
            result.append({**s, "is_numeric": is_num, "top_values": top_vals})
        return result
    except Exception:
        return []


def _load_metrics(eda_dir: str) -> dict:
    if not eda_dir:
        return {}
    path = os.path.join(eda_dir, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary(eda_dir: str) -> str:
    if not eda_dir:
        return ""
    for name in ("summary_report.md", "executive_summary.md"):
        path = os.path.join(eda_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            try:
                import markdown as md_lib
                return md_lib.markdown(raw, extensions=["tables", "fenced_code"])
            except Exception:
                return f"<pre>{raw}</pre>"
    return ""


def _list_plot_pngs(eda_dir: str, session_id: str, dataset_name: str) -> dict:
    """Return categorised PNG paths relative to AUTOEDA_ROOT for URL building."""
    if not eda_dir or not os.path.exists(eda_dir):
        return {"bivariate": [], "heatmaps": [], "distributions": [], "other": []}

    all_pngs = sorted(glob.glob(os.path.join(eda_dir, "*.png")))

    def rel(p):
        return os.path.relpath(p, AUTOEDA_ROOT).replace("\\", "/")

    bivariate = [rel(p) for p in all_pngs
                 if "bivariate_" in os.path.basename(p) or "target_interactions" in os.path.basename(p)]
    heatmaps = [rel(p) for p in all_pngs
                if any(k in os.path.basename(p) for k in ("correlation_matrix", "pairplot", "categorical_association"))]
    distributions = [rel(p) for p in all_pngs if "dist_" in os.path.basename(p)]

    used = set(bivariate) | set(heatmaps) | set(distributions)
    other = [rel(p) for p in all_pngs if rel(p) not in used]

    return {
        "bivariate": bivariate,
        "heatmaps": heatmaps,
        "distributions": distributions,
        "other": other,
    }


# ─── Views ────────────────────────────────────────────────────────────────────

def index(request):
    """Main page: controls sidebar + results area."""
    sid = _session_id(request)
    pipeline_state = pipeline_runner.get_state(sid)

    # --- Determine selected dataset ---
    selected_csv_path = request.session.get("selected_csv_path", "")
    dataset_name = (
        os.path.splitext(os.path.basename(selected_csv_path))[0]
        if selected_csv_path else ""
    )

    # --- Dataset overview metrics ---
    overview = _dataset_overview(selected_csv_path) if selected_csv_path and os.path.exists(selected_csv_path) else {}

    # --- EDA output dir ---
    eda_dir = _resolve_eda_dir(sid, dataset_name) if dataset_name else None

    # --- Column profiles ---
    profiles = _column_profiles(selected_csv_path) if selected_csv_path and os.path.exists(selected_csv_path) else []

    # --- Sanitise column names for dist PNG lookup ---
    for p in profiles:
        p["safe_name"] = re.sub(r'\W+', '_', p["column"]).strip('_')
        # Attempt to find dist PNG relative path
        if eda_dir:
            dist_path = os.path.join(eda_dir, f"dist_{p['safe_name']}.png")
            if os.path.exists(dist_path):
                p["dist_img"] = os.path.relpath(dist_path, AUTOEDA_ROOT).replace("\\", "/")
            else:
                p["dist_img"] = None
        else:
            p["dist_img"] = None

    # --- Metrics / Blueprint ---
    metrics = _load_metrics(eda_dir)
    blueprint = metrics.get("predictive_modeling_blueprint", {})
    hypothesis = metrics.get("statistical_hypothesis_tests", {})
    sig_predictors = hypothesis.pop("significant_predictors", [])
    hyp_rows = [
        {
            "feature": k,
            "test": v.get("test_name", "N/A"),
            "statistic": round(v.get("statistic", 0), 4),
            "p_value": v.get("p_value", "N/A"),
            "significant": v.get("is_statistically_significant", False),
            "interpretation": v.get("interpretation", ""),
        }
        for k, v in hypothesis.items() if isinstance(v, dict)
    ]

    # --- Executive summary ---
    summary_html = _load_summary(eda_dir)

    # --- Visual gallery ---
    plots = _list_plot_pngs(eda_dir, sid, dataset_name)

    # --- Interactive HTML report path ---
    html_report_exists = False
    html_report_rel = ""
    if eda_dir:
        hr = os.path.join(eda_dir, "eda_report.html")
        if os.path.exists(hr):
            html_report_exists = True
            html_report_rel = os.path.relpath(hr, AUTOEDA_ROOT).replace("\\", "/")

    # --- Sample datasets for the sidebar selector ---
    test_data_dir = settings.TEST_DATA_DIR
    sample_csvs = [
        os.path.basename(p)
        for p in sorted(glob.glob(os.path.join(test_data_dir, "*.csv")))
    ]

    # --- Agent question (if pipeline paused) ---
    agent_question = None
    if pipeline_state and pipeline_state.get("status") == "question":
        agent_question = pipeline_state.get("question", "")

    context = {
        "sid": sid,
        "selected_csv_path": selected_csv_path,
        "dataset_name": dataset_name,
        "overview": overview,
        "profiles": profiles,
        "metrics": metrics,
        "blueprint": blueprint,
        "hyp_rows": hyp_rows,
        "sig_predictors": sig_predictors,
        "metrics_json": json.dumps(metrics, indent=2),
        "summary_html": summary_html,
        "plots": plots,
        "html_report_exists": html_report_exists,
        "html_report_rel": html_report_rel,
        "pipeline_state": pipeline_state,
        "agent_question": agent_question,
        "sample_csvs": sample_csvs,
        "default_prompt": (
            "Perform complete exploratory analysis, type-safe missing value imputation, "
            "outlier profiling, statistical hypothesis testing, semantic bivariate graphing, "
            "and predictive blueprinting."
        ),
        "generate_summary": request.session.get("generate_summary", True),
    }
    return render(request, "eda_app/index.html", context)


@require_POST
def run_pipeline(request):
    """Launch the EDA pipeline in a background thread."""
    sid = _session_id(request)

    data_source = request.POST.get("data_source", "sample")
    generate_summary = request.POST.get("generate_summary") == "on"
    user_request = request.POST.get("user_request", "").strip() or (
        "Perform complete exploratory analysis, type-safe missing value imputation, "
        "outlier profiling, statistical hypothesis testing, semantic bivariate graphing, "
        "and predictive blueprinting."
    )

    # ---- Resolve CSV path ----
    if data_source == "upload":
        uploaded = request.FILES.get("csv_file")
        if not uploaded:
            return JsonResponse({"error": "No file uploaded."}, status=400)
        os.makedirs(settings.TEMP_UPLOADS_DIR, exist_ok=True)
        csv_path = os.path.join(settings.TEMP_UPLOADS_DIR, uploaded.name)
        with open(csv_path, "wb") as f:
            for chunk in uploaded.chunks():
                f.write(chunk)
    else:
        filename = request.POST.get("sample_file", "")
        csv_path = os.path.join(settings.TEST_DATA_DIR, filename)
        if not os.path.exists(csv_path):
            return JsonResponse({"error": f"Sample file not found: {filename}"}, status=400)

    # ---- Extract BYOK / Model Choice ----
    user_api_key = request.POST.get("openrouter_key", "").strip() or None
    user_model = request.POST.get("model_name", "").strip() or None

    # ---- Persist choices to session ----
    request.session["selected_csv_path"] = csv_path
    request.session["generate_summary"] = generate_summary
    if user_api_key:
        request.session["user_api_key"] = user_api_key
    if user_model:
        request.session["user_model"] = user_model

    workspace = _workspace_dir(sid)
    pipeline_runner.launch_pipeline(
        session_id=sid,
        data_path=csv_path,
        user_request=user_request,
        workspace_dir=workspace,
        generate_summary=generate_summary,
        conversation_history=[],
        api_key=user_api_key,
        model_name=user_model,
    )

    return JsonResponse({"status": "running", "sid": sid})


@require_GET
def pipeline_status(request):
    """Polling endpoint: returns current pipeline state as JSON."""
    sid = _session_id(request)
    state = pipeline_runner.get_state(sid)
    if not state:
        return JsonResponse({"status": "idle", "message": "", "done": False})

    status = state.get("status", "idle")
    return JsonResponse({
        "status": status,
        "message": state.get("message", ""),
        "done": status in ("done", "error"),
        "question": state.get("question") if status == "question" else None,
    })


@require_GET
def sample_datasets(request):
    """Returns JSON list of sample CSV filenames."""
    test_data_dir = settings.TEST_DATA_DIR
    files = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(test_data_dir, "*.csv")))]
    return JsonResponse({"files": files})


@require_GET
def pipeline_log(request):
    """Returns the current in-memory log buffer for the active session."""
    sid = _session_id(request)
    lines = pipeline_runner.get_log(sid)
    return JsonResponse({"lines": lines})


@require_GET
def dataset_preview(request):
    """Returns the first 50 rows of the selected dataset as JSON for table preview."""
    csv_path = request.session.get("selected_csv_path", "")
    if not csv_path or not os.path.exists(csv_path):
        return JsonResponse({"error": "No dataset selected.", "rows": [], "columns": []}, status=200)

    try:
        df = pd.read_csv(csv_path, nrows=50)
        # Determine column dtypes
        col_info = []
        for col in df.columns:
            import pandas as pd_inner
            if pd_inner.api.types.is_numeric_dtype(df[col]) and not pd_inner.api.types.is_bool_dtype(df[col]):
                dtype_label = "numeric"
            elif pd_inner.api.types.is_bool_dtype(df[col]):
                dtype_label = "bool"
            elif pd_inner.api.types.is_datetime64_any_dtype(df[col]):
                dtype_label = "datetime"
            else:
                dtype_label = "categorical"
            col_info.append({"name": col, "dtype": dtype_label, "pandas_dtype": str(df[col].dtype)})

        # Convert rows safely (NaN → None)
        rows = []
        for _, row in df.iterrows():
            rows.append({col: (None if pd.isna(v) else _safe_val(v)) for col, v in row.items()})

        total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1  # minus header
        return JsonResponse({
            "columns": col_info,
            "rows": rows,
            "total_rows": total_rows,
            "preview_rows": len(rows),
        })
    except Exception as exc:
        return JsonResponse({"error": str(exc), "rows": [], "columns": []}, status=200)


@require_POST
def reset_session(request):
    """Clears session state and pipeline runner state."""
    sid = _session_id(request)
    pipeline_runner.clear_state(sid)
    request.session.flush()
    return JsonResponse({"status": "reset"})


@require_POST
def submit_answer(request):
    """Resume pipeline after agent asked a clarifying question."""
    sid = _session_id(request)
    answer = request.POST.get("answer", "").strip()
    state = pipeline_runner.get_state(sid)

    if not state or state.get("status") != "question":
        return JsonResponse({"error": "No pending question."}, status=400)

    csv_path = request.session.get("selected_csv_path", "")
    generate_summary = request.session.get("generate_summary", True)
    user_api_key = request.session.get("user_api_key")
    user_model = request.session.get("user_model")
    conv_history = state.get("conversation_history", [])
    workspace = _workspace_dir(sid)

    pipeline_runner.launch_pipeline(
        session_id=sid,
        data_path=csv_path,
        user_request=answer,
        workspace_dir=workspace,
        generate_summary=generate_summary,
        conversation_history=conv_history,
        api_key=user_api_key,
        model_name=user_model,
    )
    return JsonResponse({"status": "running"})


def download_report(request):
    """Stream the eda_report.html as a file download."""
    sid = _session_id(request)
    dataset_name = os.path.splitext(os.path.basename(
        request.session.get("selected_csv_path", "")
    ))[0]
    eda_dir = _resolve_eda_dir(sid, dataset_name)
    if not eda_dir:
        return HttpResponse("Report not found.", status=404)

    report_path = os.path.join(eda_dir, "eda_report.html")
    if not os.path.exists(report_path):
        return HttpResponse("Report not found.", status=404)

    response = FileResponse(
        open(report_path, "rb"),
        content_type="text/html",
        as_attachment=True,
        filename=f"eda_report_{dataset_name}.html",
    )
    return response


@xframe_options_sameorigin
def serve_artifact(request, artifact_path: str):
    """Safely serve generated artifact files (PNGs, HTML) from AUTOEDA_ROOT."""
    # Prevent path traversal
    safe_path = os.path.normpath(os.path.join(AUTOEDA_ROOT, artifact_path))
    if not safe_path.startswith(AUTOEDA_ROOT):
        return HttpResponse("Forbidden.", status=403)

    if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
        return HttpResponse("Not found.", status=404)

    ext = os.path.splitext(safe_path)[1].lower()
    content_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".html": "text/html",
    }
    ct = content_type_map.get(ext, "application/octet-stream")
    return FileResponse(open(safe_path, "rb"), content_type=ct)
