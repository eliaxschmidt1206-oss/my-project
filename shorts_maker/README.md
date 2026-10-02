# Shorts Maker (OpusClip-style MCP tool)

Feed it a long video; it finds the best moments, ranks them with a 0-99 virality
score, and renders vertical 1080x1920 shorts with burned-in captions and
loudness-normalised audio. Claude can then write titles/hashtags from each
clip's returned transcript.

## Tools
| tool | what it does |
|---|---|
| `analyze_video` | duration, scene cuts, silence %, transcript excerpt |
| `find_highlights` | top clip windows: start/end, virality score, transcript |
| `create_short` | render one clip (`reframe`: blur/center, captions, trim_silence) |
| `auto_shorts` | analyze -> pick top N -> render all to `~/shorts_output` |

## Setup
```bash
sudo apt install ffmpeg          # or: brew install ffmpeg
pip install -r requirements.txt  # faster-whisper is optional but recommended
claude mcp add shorts-maker -- python /absolute/path/to/shorts_maker/server.py
```
Without faster-whisper, pass `transcript_path` (an .srt) for captions and smarter picks.

## How clips are chosen
Sliding windows scored on audio energy, speech density, hook words/questions,
cut frequency, strong first 3 seconds, minus dead air; then snapped to sentence
boundaries. Not included (vs. real OpusClip): face tracking, LLM-based scoring,
direct posting. Claude can do the LLM judging on the returned transcripts.
