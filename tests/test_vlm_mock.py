"""End-to-end VLM path against a local fake OpenAI-compatible server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rea_video.pipeline import decompose

CANNED = {
    "label_probs": {"live_action": 0.05, "ai_video": 0.05, "avatar": 0, "cg_3d": 0, "code_render": 0.8, "screen_capture": 0.1,
                    "still_motion": 0, "static_hold": 0},
    "image_source": "graphic", "evidence": ["扁平底色上的卡片滑入，文字边缘锐利"], "generator_guess": "Remotion",
    "description": "深色背景上白色卡片从左侧滑入", "shot_size": "全景", "camera_move": "固定", "style": "扁平 UI",
    "overlay_text": ["REA VIDEO"],
    "replication": {"image_prompt": "深色背景，白色圆角卡片", "video_prompt": "白色卡片从左侧滑入后停在画面中央", "video_prompt_en": "a white card slides in"},
}


class Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append(body)
        text = body["messages"][0]["content"][-1]["text"]
        payload = CANNED if "label_probs" in text else {"summary": "测试视频", "format": "知识讲解", "segments": []}
        resp = {"choices": [{"message": {"content": "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("REA_VLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("REA_VLM_API_KEY", "test")
    monkeypatch.setenv("REA_VLM_MODEL", "fake-model")
    yield Handler
    srv.shutdown()


def test_frames_media_sends_jpegs_not_video(media, tmp_path, fake_server, monkeypatch):
    monkeypatch.setenv("REA_VLM_MEDIA", "frames")
    start = len(fake_server.seen)
    b = decompose(media["code"], tmp_path / "frames", use_vlm=True, ocr=False, workers=1, stems=False, packaging=False)
    assert b["analysis"]["vlm"]["used"] and b["analysis"]["vlm"]["media"] == "frames"
    new = fake_server.seen[start:]
    assert new
    texts = []
    for body in new:
        parts = body["messages"][0]["content"]
        urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
        assert urls and all(u.startswith("data:image/jpeg") for u in urls)
        assert all(not u.startswith("data:video/") for u in urls)
        texts.append(parts[-1]["text"])
    assert any("关键帧" in t for t in texts)


def test_vlm_fusion_end_to_end(media, tmp_path, fake_server):
    start = len(fake_server.seen)
    b = decompose(media["code"], tmp_path / "o", use_vlm=True, ocr=False, workers=1, stems=False)
    assert b["analysis"]["vlm"]["used"] and not b["analysis"]["vlm"]["errors"]
    s = b["shots"][0]
    assert s["base"]["decided_by"] == "fused" and s["base"]["label"] == "code_render"
    assert s["replication"]["executor"] == "code_render_spec"
    assert b["whole_video_vlm"]["summary"] == "测试视频"
    # the shot request carried a video clip and keyframes as data URIs
    parts = fake_server.seen[start]["messages"][0]["content"]
    kinds = [p["image_url"]["url"].split(";")[0] for p in parts if p["type"] == "image_url"]
    assert kinds[0] == "data:video/mp4" and "data:image/jpeg" in kinds
    # cache hit on second run: no new requests
    n = len(fake_server.seen)
    decompose(media["code"], tmp_path / "o", use_vlm=True, ocr=False, workers=1, stems=False)
    assert len(fake_server.seen) == n


class Down(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        data = b'{"error":{"message":"model temporarily unavailable"}}'
        self.send_response(503)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_vlm_all_calls_failing_is_reported_as_not_used(media, tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), Down)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("REA_VLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("REA_VLM_API_KEY", "test")
    monkeypatch.setenv("REA_VLM_MODEL", "fake-model")
    monkeypatch.setattr("rea_video.vlm.time.sleep", lambda s: None)
    try:
        b = decompose(media["code"], tmp_path / "o", use_vlm=True, ocr=False, workers=1, stems=False)
    finally:
        srv.shutdown()
    v = b["analysis"]["vlm"]
    assert v["requested"] and not v["used"] and v["status"] == "failed" and v["errors"]
    assert "全部调用失败" in b["summary"]["vlm"]
    assert all(s["base"]["decided_by"] == "deterministic" for s in b["shots"])
    assert "全部调用失败" in (tmp_path / "o" / "report.md").read_text(encoding="utf-8")


def test_vlm_off_suggests_setup_but_still_runs(media, tmp_path):
    b = decompose(media["code"], tmp_path / "o", use_vlm=False, ocr=False, workers=1, stems=False)
    assert b["analysis"]["vlm"]["status"] == "off" and "REA_VLM_BASE_URL" in b["summary"]["vlm"]
    assert b["summary"]["production_mix"]


class FirstModelDown(Handler):
    calls = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(n))
        FirstModelDown.calls.append(body["model"])
        if body["model"] == "down-model":
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = CANNED if "label_probs" in body["messages"][0]["content"][-1]["text"] else {"summary": "x", "segments": []}
        data = json.dumps({"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_vlm_falls_back_and_sticks_to_working_model(media, tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FirstModelDown)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("REA_VLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.setenv("REA_VLM_API_KEY", "test")
    monkeypatch.setenv("REA_VLM_MODEL", "down-model,good-model")
    monkeypatch.setattr("rea_video.vlm.time.sleep", lambda s: None)
    try:
        b = decompose(media["code"], tmp_path / "o", use_vlm=True, ocr=False, workers=1, stems=False)
    finally:
        srv.shutdown()
    assert b["analysis"]["vlm"]["status"] == "ok"
    # the dead model is tried on the first request only
    assert FirstModelDown.calls.count("down-model") == 2 and FirstModelDown.calls[-1] == "good-model"
