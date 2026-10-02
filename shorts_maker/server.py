"""Shorts Maker - an MCP server that reviews a long video and cuts it into
vertical (9:16) short-form clips for YouTube Shorts / TikTok / Reels.

Tools exposed to Claude:
  analyze_video   probe + scenes + silence + loudness (+ transcript)
  find_highlights rank the best clip windows
  create_short    render one clip (cut, reframe, captions, loudness)
  auto_shorts     analyze -> pick top N -> render all

Requires: ffmpeg/ffprobe on PATH. Optional: faster-whisper for transcripts.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("shorts-maker")

OUT_DIR = Path.home() / "shorts_output"
HOOK_WORDS = {
    "secret", "never", "always", "mistake", "truth", "why", "how", "wait",
    "crazy", "insane", "shocking", "actually", "biggest", "worst", "best",
    "nobody", "everyone", "stop", "warning", "free", "proof", "watch",
}


# --------------------------------------------------------------------- utils
def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-1500:]}")
    return p


def _need_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. Install ffmpeg first.")


def _probe(path: str) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    out = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                "-of", "json", path]).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise ValueError("No video stream found")
    return {
        "duration": float(info["format"]["duration"]),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def _scene_changes(path: str, threshold: float = 0.35) -> list[float]:
    p = _run(["ffmpeg", "-hide_banner", "-i", path, "-an", "-vf",
              f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
             check=False)
    return [float(m) for m in re.findall(r"pts_time:([\d.]+)", p.stderr)]


def _silences(path: str, noise_db: int = -35, min_len: float = 0.6) -> list[tuple[float, float]]:
    p = _run(["ffmpeg", "-hide_banner", "-i", path, "-vn", "-af",
              f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
             check=False)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    if len(starts) > len(ends):  # silence runs to the end
        ends.append(float("inf"))
    return list(zip(starts, ends))


def _loudness_per_second(path: str) -> list[float]:
    """RMS level (dB) for each second of audio."""
    p = _run(["ffmpeg", "-hide_banner", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
              "-af", "asetnsamples=n=16000:p=0,astats=metadata=1:reset=1,"
                     "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
              "-f", "null", "-"], check=False)
    vals = []
    for m in re.findall(r"RMS_level=(-?[\d.]+|-inf)", p.stdout):
        vals.append(-90.0 if m == "-inf" else max(-90.0, float(m)))
    return vals


# ---------------------------------------------------------------- transcript
def _parse_srt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    ts = r"(\d+):(\d+):(\d+)[,.](\d+)"
    words = []
    for block in re.split(r"\n\s*\n", text.strip()):
        m = re.search(f"{ts}\\s*-->\\s*{ts}", block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = block[m.end():].strip().replace("\n", " ")
        toks = body.split()
        for i, t in enumerate(toks):  # spread words evenly across the cue
            w0 = s + (e - s) * i / len(toks)
            w1 = s + (e - s) * (i + 1) / len(toks)
            words.append({"w": t, "start": w0, "end": w1})
    return words


def _transcribe(path: str, model_size: str = "base") -> list[dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    model = WhisperModel(model_size, compute_type="int8")
    segs, _ = model.transcribe(path, word_timestamps=True, vad_filter=True)
    return [{"w": w.word.strip(), "start": w.start, "end": w.end}
            for s in segs for w in (s.words or []) if w.word.strip()]


def _get_words(path: str, transcript_path: str | None, model_size: str) -> list[dict]:
    if transcript_path:
        return _parse_srt(transcript_path)
    return _transcribe(path, model_size)


# ------------------------------------------------------------------- scoring
def _norm(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


def _rank_windows(duration, loud, scenes, silences, words,
                  min_len, max_len, top_n) -> list[dict]:
    n = int(duration)
    loud = (loud + [-90.0] * n)[:n]
    loud_n = _norm(loud)
    scene_hist = [0.0] * n
    for t in scenes:
        if int(t) < n:
            scene_hist[int(t)] += 1
    word_hist = [0.0] * n
    hook_hist = [0.0] * n
    for w in words:
        i = int(w["start"])
        if i < n:
            word_hist[i] += 1
            if re.sub(r"\W", "", w["w"].lower()) in HOOK_WORDS or "?" in w["w"]:
                hook_hist[i] += 1
    silent = [0.0] * n
    for s, e in silences:
        for i in range(int(s), min(n, int(math.ceil(e)) if e != float("inf") else n)):
            silent[i] = 1.0
    wn, sn, hn = _norm(word_hist), _norm(scene_hist), _norm(hook_hist)

    length = int((min_len + max_len) / 2)
    if n <= length:
        length = n
    # prefix sums for O(1) window means
    def pref(a):
        p = [0.0]
        for x in a:
            p.append(p[-1] + x)
        return p
    P = {k: pref(v) for k, v in dict(l=loud_n, w=wn, s=sn, h=hn, q=silent).items()}
    mean = lambda k, a, b: (P[k][b] - P[k][a]) / max(1, b - a)

    cands = []
    for a in range(0, max(1, n - length + 1), 3):
        b = min(n, a + length)
        score = (0.30 * mean("l", a, b) + 0.30 * mean("w", a, b)
                 + 0.15 * mean("s", a, b) + 0.25 * mean("h", a, b)
                 - 0.40 * mean("q", a, b))
        # hook strength: energy in the first 3 seconds
        score += 0.15 * mean("l", a, min(b, a + 3)) + 0.10 * mean("h", a, min(b, a + 5))
        cands.append((score, a, b))
    cands.sort(reverse=True)

    picked = []
    for score, a, b in cands:
        if all(b <= pa or a >= pb for _, pa, pb in picked):
            picked.append((score, a, b))
        if len(picked) >= top_n:
            break

    out = []
    for score, a, b in picked:
        a, b = _snap(a, b, words, min_len, max_len, duration)
        text = " ".join(w["w"] for w in words if a <= w["start"] < b)
        out.append({"start": round(a, 2), "end": round(b, 2),
                    "duration": round(b - a, 1), "score": round(score, 3),
                    "virality": max(1, min(99, round(100 * score / 1.1))),
                    "preview": text[:160], "transcript": text})
    return sorted(out, key=lambda c: -c["score"])


def _snap(a, b, words, min_len, max_len, duration):
    """Start on a sentence start and end on a sentence end when we have words."""
    if words:
        starts = [w["start"] for i, w in enumerate(words)
                  if i == 0 or re.search(r"[.!?]$", words[i - 1]["w"])]
        ends = [w["end"] for w in words if re.search(r"[.!?]$", w["w"])]
        a2 = min(starts, key=lambda t: abs(t - a), default=a)
        e2 = [t for t in ends if min_len <= t - a2 <= max_len]
        b = min(e2, key=lambda t: abs(t - b)) if e2 else min(b, a2 + max_len)
        a = a2
    return max(0.0, a), min(duration, b)


# ----------------------------------------------------------------- rendering
def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _write_ass(words, start, end, path, w, h, style="bold") -> bool:
    ws = [x for x in words if start <= x["start"] < end]
    if not ws:
        return False
    size = int(h * 0.050)
    margin_v = int(h * 0.22)
    colour = "&H0000FFFF" if style == "bold" else "&H00FFFFFF"  # yellow / white
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Default,DejaVu Sans,{size},{colour},&H00FFFFFF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,{max(3, size // 12)},1,2,60,60,{margin_v},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    for i in range(0, len(ws), 3):  # 3-word punchy chunks
        chunk = ws[i:i + 3]
        t0 = chunk[0]["start"] - start
        t1 = (ws[i + 3]["start"] if i + 3 < len(ws) else chunk[-1]["end"]) - start
        txt = " ".join(c["w"] for c in chunk).upper().replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{_ass_time(max(0, t0))},{_ass_time(max(t0 + .2, t1))},"
                     f"Default,,0,0,0,,{txt}")
    Path(path).write_text(head + "\n".join(lines), encoding="utf-8")
    return True


def _render(src, start, end, out, *, reframe, words, captions, trim_silence,
            out_w=1080, out_h=1920) -> str:
    info = _probe(src)
    dur = end - start
    if dur <= 0:
        raise ValueError("end must be greater than start")
    tmp = Path(tempfile.mkdtemp(prefix="shorts_"))
    ass = tmp / "cap.ass"
    has_caps = captions and _write_ass(words, start, end, ass, out_w, out_h)

    if reframe == "center":
        vf = f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
    else:  # "blur": sharp video on a blurred, zoomed copy of itself
        vf = (f"split[a][b];[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
              f"crop={out_w}:{out_h},boxblur=30:5[bg];"
              f"[b]scale={out_w}:-2:force_original_aspect_ratio=decrease[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if has_caps:
        esc = str(ass).replace("\\", "/").replace(":", "\\:")
        vf += f",ass='{esc}'"
    vf += ",format=yuv420p"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", src,
           "-vf", vf]
    if info["has_audio"]:
        af = "loudnorm=I=-14:TP=-1.5:LRA=11"
        if trim_silence:
            af = ("silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB,"
                  + af)
        cmd += ["-af", af, "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", "30",
            "-movflags", "+faststart", str(out)]
    try:
        _run(cmd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return str(out)


# --------------------------------------------------------------------- tools
@mcp.tool()
def analyze_video(video_path: str, transcript_path: str = "",
                  whisper_model: str = "base") -> dict:
    """Review a video: duration, resolution, scene cuts, silence, loudness, transcript.

    Args:
        video_path: Absolute path to the source video.
        transcript_path: Optional .srt file; if empty, faster-whisper is used when installed.
        whisper_model: faster-whisper model size (tiny/base/small/medium/large-v3).
    """
    _need_ffmpeg()
    info = _probe(video_path)
    scenes = _scene_changes(video_path)
    sil = _silences(video_path) if info["has_audio"] else []
    words = _get_words(video_path, transcript_path or None, whisper_model)
    dur = info["duration"]
    silent_total = sum(min(e, dur) - s for s, e in sil)
    return {
        **info,
        "scene_changes": len(scenes),
        "cuts_per_minute": round(len(scenes) / (dur / 60), 1) if dur else 0,
        "silence_pct": round(100 * silent_total / dur, 1) if dur else 0,
        "word_count": len(words),
        "transcript_available": bool(words),
        "transcript_excerpt": " ".join(w["w"] for w in words[:120]),
        "note": "" if words else
        "No transcript: install faster-whisper or pass an .srt for captions and smarter clip picks.",
    }


@mcp.tool()
def find_highlights(video_path: str, count: int = 5, min_seconds: int = 20,
                    max_seconds: int = 60, transcript_path: str = "",
                    whisper_model: str = "base") -> list[dict]:
    """Rank the best short-form clip windows (energy, speech density, hook words, cuts).

    Returns clips with start/end seconds, score and a text preview. Use these
    with create_short, or review them with the user first.
    """
    _need_ffmpeg()
    info = _probe(video_path)
    words = _get_words(video_path, transcript_path or None, whisper_model)
    loud = _loudness_per_second(video_path) if info["has_audio"] else []
    return _rank_windows(info["duration"], loud, _scene_changes(video_path),
                         _silences(video_path) if info["has_audio"] else [], words,
                         min_seconds, max_seconds, count)


@mcp.tool()
def create_short(video_path: str, start: float, end: float, output_name: str = "",
                 reframe: str = "blur", captions: bool = True,
                 trim_silence: bool = False, transcript_path: str = "",
                 whisper_model: str = "base") -> dict:
    """Render one vertical 1080x1920 short from [start, end] seconds.

    Args:
        reframe: "blur" (full frame over blurred background) or "center" (crop to fill).
        captions: Burn in big word-chunk captions (needs transcript or faster-whisper).
        trim_silence: Remove dead air > 0.5s (tightens pacing; may drift caption sync).
    """
    _need_ffmpeg()
    if reframe not in ("blur", "center"):
        raise ValueError('reframe must be "blur" or "center"')
    if end - start > 180:
        raise ValueError("Shorts must be <= 180s; pick a shorter range.")
    words = _get_words(video_path, transcript_path or None, whisper_model) if captions else []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = output_name or f"{Path(video_path).stem}_{int(start)}-{int(end)}.mp4"
    out = OUT_DIR / (name if name.endswith(".mp4") else name + ".mp4")
    _render(video_path, start, end, out, reframe=reframe, words=words,
            captions=captions, trim_silence=trim_silence)
    return {"output": str(out), "duration": round(end - start, 1),
            "captions_burned": bool(words and captions)}


@mcp.tool()
def auto_shorts(video_path: str, count: int = 3, min_seconds: int = 20,
                max_seconds: int = 60, reframe: str = "blur", captions: bool = True,
                trim_silence: bool = False, transcript_path: str = "",
                whisper_model: str = "base") -> dict:
    """End to end: analyze the video, pick the top `count` moments, render each as a short."""
    _need_ffmpeg()
    words = _get_words(video_path, transcript_path or None, whisper_model)
    info = _probe(video_path)
    clips = _rank_windows(info["duration"],
                          _loudness_per_second(video_path) if info["has_audio"] else [],
                          _scene_changes(video_path),
                          _silences(video_path) if info["has_audio"] else [], words,
                          min_seconds, max_seconds, count)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, c in enumerate(clips, 1):
        out = OUT_DIR / f"{Path(video_path).stem}_short{i}.mp4"
        _render(video_path, c["start"], c["end"], out, reframe=reframe, words=words,
                captions=captions, trim_silence=trim_silence)
        results.append({**c, "output": str(out)})
    return {"shorts": results, "output_dir": str(OUT_DIR)}


if __name__ == "__main__":
    mcp.run()
