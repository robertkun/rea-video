import json
import time
from pathlib import Path

import pytest

from rea_video.pipeline import whole_keyframes
from rea_video.web import _safe_file, resolve_deepseek, summarize_breakdown


def test_whole_keyframes_samples_evenly(tmp_path):
    shots = []
    for i in range(10):
        path = tmp_path / f"f{i}.jpg"
        path.write_bytes(b"x")
        shots.append({"assets": {"first_frame": path.name}})
    got = [p.name for p, mime in whole_keyframes(shots, tmp_path, limit=4)]
    assert got == ["f0.jpg", "f3.jpg", "f6.jpg", "f9.jpg"]
    assert all(m == "image/jpeg" for _, m in whole_keyframes(shots, tmp_path, limit=4))


def test_resolve_deepseek_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("REA_VLM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        resolve_deepseek("")
    cfg = resolve_deepseek("sk-abc", "https://api.deepseek.com/v1/chat/completions/", "deepseek-flash")
    assert cfg["base_url"] == "https://api.deepseek.com/v1"
    assert cfg["model"] == "deepseek-flash"
    assert cfg["api_key"] == "sk-abc"


def test_safe_file_blocks_escape(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (root / "ok.txt").write_text("a", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    assert _safe_file(root, "ok.txt").name == "ok.txt"
    assert _safe_file(root, "../secret.txt") is None


def test_summary_omits_nothing_secret():
    view = summarize_breakdown("abc", {"summary": {"verdict": "代码", "shots": 1, "production_mix": []}, "shots": [], "analysis": {"vlm": {"status": "ok"}}})
    assert view["report"] == "/runs/abc/report.html"
    assert "api_key" not in view


def test_job_api_hides_key_and_runs(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from rea_video.web import create_app

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("REA_VLM_API_KEY", raising=False)
    secret = "sk-test-SECRET-991"

    def fake_decompose(video, out, **kwargs):
        assert kwargs["use_vlm"] is True
        assert kwargs["max_shots_vlm"] == 40
        import os
        assert os.environ["REA_VLM_API_KEY"] == secret
        assert os.environ["REA_VLM_MODEL"] == "deepseek-flash"
        assert os.environ["REA_VLM_MEDIA"] == "frames"
        assert "deepseek.com" in os.environ["REA_VLM_BASE_URL"]
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.html").write_text("<html>报告</html>", encoding="utf-8")
        (out / "shots").mkdir(exist_ok=True)
        (out / "shots" / "shot_001_first.jpg").write_bytes(b"\xff\xd8")
        b = {
            "summary": {
                "verdict": "代码/模板渲染", "confidence": 0.9, "shots": 1, "duration": 1.2,
                "production_mix": [{"label": "code_render", "label_zh": "代码/模板渲染", "ratio": 1, "seconds": 1.2}],
                "vlm": "VLM 已生效（deepseek-flash @ api.deepseek.com）",
            },
            "analysis": {"vlm": {"status": "ok"}},
            "shots": [{
                "id": "shot_001", "index": 1, "start": 0, "end": 1.2,
                "base": {"label_zh": "代码/模板渲染", "confidence": 0.9, "evidence": ["卡片"]},
                "vlm": {"description": "白色卡片滑入"},
                "assets": {"first_frame": "shots/shot_001_first.jpg"},
            }],
        }
        (out / "breakdown.json").write_text(json.dumps(b), encoding="utf-8")
        return b

    monkeypatch.setattr("rea_video.pipeline.decompose", fake_decompose)
    client = TestClient(create_app(tmp_path))
    missing = client.post("/api/jobs", files={"file": ("a.mp4", b"not-a-video", "video/mp4")})
    assert missing.status_code == 400
    page = client.get("/")
    assert page.status_code == 200 and "DeepSeek" in page.text

    res = client.post(
        "/api/jobs",
        data={"api_key": secret, "model": "deepseek-flash", "base_url": "https://api.deepseek.com", "max_shots": "40"},
        files={"file": ("clip.mp4", b"not-a-video", "video/mp4")},
    )
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]
    body = {}
    for _ in range(50):
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert body["status"] == "done", body
    dumped = json.dumps(body, ensure_ascii=False)
    assert secret not in dumped
    assert body["result"]["verdict"] == "代码/模板渲染"
    assert body["result"]["shots"][0]["description"] == "白色卡片滑入"
    report = client.get(body["result"]["report"])
    assert report.status_code == 200 and "报告" in report.text
    assert client.get(f"/runs/{job_id}/../secret.txt").status_code == 404
