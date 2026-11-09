"""Local page: upload a video, send keyframes to DeepSeek, show the breakdown.

DeepSeek's vision model accepts images, not video parts, so this always sets
REA_VLM_MEDIA=frames. The API key stays in this process and is never written
into the report or the job status.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import threading
import uuid
import webbrowser
from pathlib import Path

from .util import LOG

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
MAX_UPLOAD = 512 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
_STATIC = Path(__file__).resolve().parent / "web_static" / "index.html"
_ENV_KEYS = ("REA_VLM_BASE_URL", "REA_VLM_API_KEY", "REA_VLM_MODEL", "REA_VLM_MEDIA", "REA_VLM_TIMEOUT")


def env_deepseek_key() -> str:
    if os.getenv("DEEPSEEK_API_KEY", "").strip():
        return os.getenv("DEEPSEEK_API_KEY", "").strip()
    base = os.getenv("REA_VLM_BASE_URL", "")
    if "deepseek" in base.lower() and os.getenv("REA_VLM_API_KEY", "").strip():
        return os.getenv("REA_VLM_API_KEY", "").strip()
    return ""


def resolve_deepseek(api_key: str = "", base_url: str = "", model: str = "") -> dict:
    key = (api_key or env_deepseek_key()).strip()
    if not key:
        raise ValueError("请先填写 DeepSeek API Key，或设置环境变量 DEEPSEEK_API_KEY")
    base = (base_url or DEEPSEEK_BASE).strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    if not (base.startswith("https://") or base.startswith("http://")):
        raise ValueError("接口地址需要以 http:// 或 https:// 开头")
    name = (model or DEEPSEEK_MODEL).strip() or DEEPSEEK_MODEL
    return {"base_url": base, "api_key": key, "model": name}


def summarize_breakdown(job_id: str, b: dict) -> dict:
    shots = []
    for s in b.get("shots") or []:
        base = s.get("base") or {}
        assets = s.get("assets") or {}
        thumb = assets.get("first_frame")
        shots.append({
            "id": s.get("id"),
            "index": s.get("index"),
            "start": s.get("start"),
            "end": s.get("end"),
            "label": base.get("label_zh") or base.get("label") or "",
            "confidence": base.get("confidence"),
            "evidence": list(base.get("evidence") or [])[:4],
            "description": ((s.get("vlm") or {}).get("description") or ""),
            "thumb": f"/runs/{job_id}/{thumb}" if thumb else None,
        })
    summary = b.get("summary") or {}
    vlm = (b.get("analysis") or {}).get("vlm") or {}
    return {
        "verdict": summary.get("verdict") or "",
        "confidence": summary.get("confidence"),
        "shot_count": summary.get("shots", len(shots)),
        "duration": summary.get("duration"),
        "mix": summary.get("production_mix") or [],
        "vlm": summary.get("vlm") or "",
        "vlm_status": vlm.get("status") or "",
        "report": f"/runs/{job_id}/report.html",
        "shots": shots,
    }


def _push_env(cfg: dict) -> dict:
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["REA_VLM_BASE_URL"] = cfg["base_url"]
    os.environ["REA_VLM_API_KEY"] = cfg["api_key"]
    os.environ["REA_VLM_MODEL"] = cfg["model"]
    os.environ["REA_VLM_MEDIA"] = "frames"
    os.environ["REA_VLM_TIMEOUT"] = saved["REA_VLM_TIMEOUT"] or "180"
    return saved


def _pop_env(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _JobLog(logging.Handler):
    def __init__(self, job: "Job"):
        super().__init__(level=logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "api_key" in msg.lower() or "authorization" in msg.lower():
            return
        self.job.logs.append(msg)
        if len(self.job.logs) > 200:
            self.job.logs = self.job.logs[-200:]


class Job:
    def __init__(self, job_id: str, out: Path, filename: str, cfg: dict, *, lang: str, max_shots: int):
        self.id = job_id
        self.out = out
        self.filename = filename
        self.cfg = cfg
        self.lang = lang
        self.max_shots = max_shots
        self.status = "queued"
        self.error = ""
        self.logs: list[str] = []
        self.result: dict | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "logs": self.logs[-80:],
            "result": self.result,
        }


class Hub:
    def __init__(self, runs: Path):
        self.runs = runs
        self.runs.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="rea-video-web", daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> None:
        self.jobs[job.id] = job
        self.queue.put(job)

    def _loop(self) -> None:
        while True:
            job = self.queue.get()
            try:
                self._run(job)
            except Exception as exc:  # last resort; _run already records failures
                job.status = "error"
                job.error = str(exc)[:500]
            finally:
                self.queue.task_done()

    def _run(self, job: Job) -> None:
        from .pipeline import decompose

        job.status = "running"
        handler = _JobLog(job)
        LOG.addHandler(handler)
        saved = _push_env(job.cfg)
        try:
            video = job.out / "input" / f"source{Path(job.filename).suffix.lower()}"
            breakdown = decompose(
                video, job.out, use_vlm=True, whole_vlm=True, workers=2,
                max_shots_vlm=job.max_shots, lang=job.lang,
            )
            job.result = summarize_breakdown(job.id, breakdown)
            job.status = "done"
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)[:500]
            LOG.warning("web job %s failed: %s", job.id, job.error)
        finally:
            _pop_env(saved)
            LOG.removeHandler(handler)


def _safe_file(root: Path, rel: str) -> Path | None:
    try:
        target = (root / rel).resolve()
        target.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    return target


def create_app(runs: Path):
    try:
        from fastapi import FastAPI, File, Form, HTTPException
        from fastapi.responses import FileResponse, HTMLResponse
    except ImportError as exc:
        raise SystemExit("缺少网页依赖，请先运行：pip install -e \".[web]\"") from exc

    hub = Hub(Path(runs))
    app = FastAPI(title="rea-video", docs_url=None, redoc_url=None)
    app.state.hub = hub

    @app.get("/", response_class=HTMLResponse)
    def index():
        if not _STATIC.is_file():
            raise HTTPException(500, "找不到页面文件")
        return HTMLResponse(_STATIC.read_text(encoding="utf-8"))

    @app.get("/api/config")
    def config():
        return {
            "model": DEEPSEEK_MODEL,
            "base_url": DEEPSEEK_BASE,
            "has_key": bool(env_deepseek_key()),
            "media": "frames",
        }

    @app.post("/api/jobs")
    async def create_job(
        file=File(...),
        api_key: str = Form(""),
        model: str = Form(""),
        base_url: str = Form(""),
        lang: str = Form("zh"),
        max_shots: int = Form(40),
    ):
        try:
            cfg = resolve_deepseek(api_key, base_url, model)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if lang not in ("zh", "en"):
            raise HTTPException(400, "报告语言只支持 zh 或 en")
        if not 1 <= int(max_shots) <= 60:
            raise HTTPException(400, "看图镜头数需要在 1 到 60 之间")
        name = Path(file.filename or "video.mp4").name
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(400, "请上传 mp4、mov、webm、mkv、m4v 或 avi")
        job_id = uuid.uuid4().hex[:12]
        out = hub.runs / job_id
        src_dir = out / "input"
        src_dir.mkdir(parents=True, exist_ok=True)
        dest = src_dir / f"source{suffix}"
        size = 0
        try:
            with dest.open("wb") as fh:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "视频超过 512MB")
                    fh.write(chunk)
        except HTTPException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        if size == 0:
            shutil.rmtree(out, ignore_errors=True)
            raise HTTPException(400, "视频文件是空的")
        job = Job(job_id, out, name, cfg, lang=lang, max_shots=int(max_shots))
        hub.submit(job)
        return {"id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = hub.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "没有这个任务")
        return job.public()

    @app.get("/runs/{job_id}/{path:path}")
    def run_file(job_id: str, path: str):
        root = (hub.runs / job_id).resolve()
        if not root.is_dir() or (hub.runs.resolve() not in root.parents and root != hub.runs.resolve()):
            raise HTTPException(404, "没有这个结果")
        target = _safe_file(root, path)
        if target is None:
            raise HTTPException(404, "没有这个文件")
        return FileResponse(target)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True, runs_dir: str | Path | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("缺少网页依赖，请先运行：pip install -e \".[web]\"") from exc
    app = create_app(Path(runs_dir) if runs_dir else Path("out/web"))
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    LOG.info("rea-video web %s", url)
    uvicorn.run(app, host=host, port=port, log_level="info")
