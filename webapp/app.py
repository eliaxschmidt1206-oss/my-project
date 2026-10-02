"""ClipForge web app: paste a YouTube link (or upload a video) and get back
several vertical short-form clips of the best moments, with captions.

Run:  uvicorn webapp.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shorts_maker"))
import server as core  # noqa: E402  (the clip-finding / rendering engine)

DATA = Path(os.environ.get("CLIPFORGE_DATA", "/tmp/clipforge"))
MAX_MINUTES = int(os.environ.get("CLIPFORGE_MAX_MINUTES", "120"))
MAX_UPLOAD = int(os.environ.get("CLIPFORGE_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
KEEP_SECONDS = 24 * 3600
YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}

DATA.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="ClipForge")
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("CLIPFORGE_WORKERS", "1")))
jobs: dict[str, dict] = {}


def _set(job_id: str, **kw) -> None:
    jobs[job_id].update(kw)


def _download(url: str, dest: Path, job_id: str) -> Path:
    import yt_dlp

    def hook(d):
        if d["status"] == "downloading" and d.get("total_bytes"):
            _set(job_id, progress=round(10 * d["downloaded_bytes"] / d["total_bytes"]))

    opts = {
        "outtmpl": str(dest / "source.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "progress_hooks": [hook],
        "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {MAX_MINUTES * 60}"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        _set(job_id, title=info.get("title", ""))
    files = list(dest.glob("source.*"))
    if not files:
        raise RuntimeError(f"Video is longer than {MAX_MINUTES} minutes or unavailable.")
    return files[0]


def _process(job_id: str, url: str | None, src: Path | None, opts: dict) -> None:
    jdir = DATA / job_id
    try:
        if url:
            _set(job_id, status="downloading", progress=2)
            src = _download(url, jdir, job_id)
        info = core._probe(str(src))
        if info["duration"] > MAX_MINUTES * 60:
            raise RuntimeError(f"Video is longer than {MAX_MINUTES} minutes.")
        _set(job_id, status="analyzing", progress=12)
        words = core._get_words(str(src), None, "base") if opts["captions"] else []
        has_a = info["has_audio"]
        _set(job_id, progress=25)
        loud = core._loudness_per_second(str(src)) if has_a else []
        scenes = core._scene_changes(str(src))
        sil = core._silences(str(src)) if has_a else []
        clips = core._rank_windows(info["duration"], loud, scenes, sil, words,
                                   opts["min_s"], opts["max_s"], opts["count"])
        if not clips:
            raise RuntimeError("Couldn't find any good moments in this video.")
        _set(job_id, status="rendering", progress=35)
        results = []
        for i, c in enumerate(clips, 1):
            name = f"short{i}.mp4"
            core._render(str(src), c["start"], c["end"], jdir / name, reframe=opts["reframe"],
                         words=words, captions=opts["captions"], trim_silence=False)
            results.append({**c, "file": f"/files/{job_id}/{name}"})
            _set(job_id, progress=35 + round(65 * i / len(clips)), clips=list(results))
        _set(job_id, status="done", progress=100, captions=bool(words))
    except Exception as e:  # surface a readable message to the UI
        _set(job_id, status="error", error=str(e)[:400])
    finally:
        if src and src.exists():
            src.unlink(missing_ok=True)  # don't keep the (large) source video


@app.post("/api/jobs")
async def create_job(url: str = Form(""), file: UploadFile | None = File(None),
                     count: int = Form(5), min_s: int = Form(20), max_s: int = Form(60),
                     reframe: str = Form("blur"), captions: bool = Form(True)):
    count, min_s, max_s = max(1, min(count, 10)), max(10, min_s), min(120, max_s)
    if min_s >= max_s or reframe not in ("blur", "center"):
        raise HTTPException(400, "Invalid options.")
    job_id = uuid.uuid4().hex
    jdir = DATA / job_id
    jdir.mkdir(parents=True)
    src = None
    if url:
        host = (urlparse(url).hostname or "").lower()
        if urlparse(url).scheme not in ("http", "https") or host not in YT_HOSTS:
            shutil.rmtree(jdir)
            raise HTTPException(400, "Please paste a YouTube link.")
    elif file and file.filename:
        src = jdir / ("upload" + Path(file.filename).suffix.lower()[:8])
        size = 0
        with open(src, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    f.close()
                    shutil.rmtree(jdir)
                    raise HTTPException(413, "File too large.")
                f.write(chunk)
    else:
        shutil.rmtree(jdir)
        raise HTTPException(400, "Provide a YouTube link or a video file.")
    jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0, "clips": [],
                    "created": time.time()}
    pool.submit(_process, job_id, url or None, src,
                dict(count=count, min_s=min_s, max_s=max_s, reframe=reframe, captions=captions))
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job (it may have expired).")
    return jobs[job_id]


@app.get("/files/{job_id}/{name}")
def get_file(job_id: str, name: str, download: bool = False):
    if job_id not in jobs or not name.startswith("short") or not name.endswith(".mp4") \
            or "/" in name:
        raise HTTPException(404)
    path = DATA / job_id / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4",
                        filename=name if download else None)


def _janitor() -> None:
    while True:
        time.sleep(600)
        for jid, j in list(jobs.items()):
            if time.time() - j["created"] > KEEP_SECONDS and j["status"] in ("done", "error"):
                shutil.rmtree(DATA / jid, ignore_errors=True)
                jobs.pop(jid, None)


threading.Thread(target=_janitor, daemon=True).start()
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
