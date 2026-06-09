"""ffmpeg helpers for the news-video pipeline.

Uses imageio-ffmpeg's bundled static ffmpeg binary so no system package is required
(the deploy step only runs ``pip install``). All functions are blocking and are meant
to run inside the background generation thread (never the HTTP request).
"""
import logging
import os
import subprocess
import tempfile

import imageio_ffmpeg

from GenBox.newsvideo import config

logger = logging.getLogger("GenBoxVideo.mux")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_W, _H = (int(x) for x in config.VIDEO_SIZE.split("x"))

# Uniform encode params so the concat demuxer can stream-copy the normalized clips.
_FPS = 30
_AR = "44100"
_AC = "2"


def _run(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, capture_output=True)


def _has_audio(path: str) -> bool:
    """Detect an audio stream by parsing ffmpeg's stderr stream report.

    ``ffmpeg -i <path>`` with no output exits non-zero but prints stream metadata to
    stderr; we look for an "Audio:" stream line. (imageio-ffmpeg ships ffmpeg, not
    ffprobe, so we avoid ffprobe here.)
    """
    proc = subprocess.run([FFMPEG, "-hide_banner", "-i", path], capture_output=True)
    return b"Audio:" in (proc.stderr or b"")


def extract_last_frame(clip_bytes: bytes) -> bytes:
    """Return the clip's final frame as PNG bytes resized to exactly VIDEO_SIZE.

    The result is used as a Sora ``input_reference`` (which must match ``size``).
    """
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mp4")
        out = os.path.join(d, "frame.png")
        with open(src, "wb") as fh:
            fh.write(clip_bytes)
        # Seek to the last ~1s, then grab a single frame.
        _run([FFMPEG, "-y", "-sseof", "-1", "-i", src,
              "-frames:v", "1", "-vf", f"scale={_W}:{_H}", out])
        with open(out, "rb") as fh:
            return fh.read()


def _normalize_clip(src: str, dst: str) -> None:
    """Re-encode one clip to uniform 1280x720 H.264 / AAC, adding silent audio if the
    clip has none (the concat demuxer needs every clip to share stream layout)."""
    vf = f"scale={_W}:{_H},setsar=1,fps={_FPS}"
    if _has_audio(src):
        args = [FFMPEG, "-y", "-i", src,
                "-vf", vf,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(_FPS),
                "-c:a", "aac", "-ar", _AR, "-ac", _AC,
                dst]
    else:
        args = [FFMPEG, "-y", "-i", src,
                "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={_AR}",
                "-vf", vf,
                "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(_FPS),
                "-c:a", "aac", "-ar", _AR, "-ac", _AC,
                dst]
    _run(args)


def concat_audio_with_gaps(segments: list, gap_seconds: float = 0.5,
                           lead_seconds: float = 0.5, trail_seconds: float = 0.0,
                           fmt: str = "mp3") -> bytes:
    """Concatenate audio byte-segments into one file, inserting silence between/around them.

    Playback order is: ``lead`` silence, segment[0], ``gap`` silence, segment[1], ... ,
    ``trail`` silence. Each segment and each silence is normalized to a common stereo
    44.1kHz format and concatenated via ffmpeg's ``concat`` filter (re-encoded), so the
    inputs don't need matching codec parameters. Returns the merged audio bytes.
    """
    segs = [s for s in segments if s]
    if not segs:
        raise ValueError("concat_audio_with_gaps: no audio segments")

    sr = int(_AR)
    with tempfile.TemporaryDirectory() as d:
        input_args = []
        n_inputs = 0

        # Segment file inputs (ffmpeg input indices 0..n-1).
        seg_idx = []
        for i, s in enumerate(segs):
            p = os.path.join(d, f"seg{i}.{fmt}")
            with open(p, "wb") as fh:
                fh.write(s)
            input_args += ["-i", p]
            seg_idx.append(n_inputs)
            n_inputs += 1

        # Build the playback order, adding one lavfi silence input per gap as needed.
        order = []

        def add_silence(dur):
            nonlocal n_inputs
            if not dur or dur <= 0:
                return
            input_args.extend(["-f", "lavfi", "-t", f"{dur:g}",
                               "-i", f"anullsrc=r={sr}:cl=stereo"])
            order.append(n_inputs)
            n_inputs += 1

        add_silence(lead_seconds)
        for k, si in enumerate(seg_idx):
            order.append(si)
            if k < len(seg_idx) - 1:
                add_silence(gap_seconds)
        add_silence(trail_seconds)

        # Normalize each part to a common format, then concat in playback order.
        filt = []
        labels = []
        for j, inp in enumerate(order):
            lbl = f"a{j}"
            filt.append(f"[{inp}:a]aresample={sr},aformat=channel_layouts=stereo[{lbl}]")
            labels.append(f"[{lbl}]")
        filt.append(f"{''.join(labels)}concat=n={len(order)}:v=0:a=1[out]")

        out = os.path.join(d, f"out.{fmt}")
        _run([FFMPEG, "-y", *input_args,
              "-filter_complex", ";".join(filt), "-map", "[out]", out])
        with open(out, "rb") as fh:
            return fh.read()


def concat_clips(clip_bytes_list: list) -> bytes:
    """Normalize every clip then concatenate into one MP4 (faststart). Returns MP4 bytes."""
    if not clip_bytes_list:
        raise ValueError("concat_clips: no clips provided")
    with tempfile.TemporaryDirectory() as d:
        norm_paths = []
        for i, clip in enumerate(clip_bytes_list):
            src = os.path.join(d, f"src{i}.mp4")
            dst = os.path.join(d, f"norm{i}.mp4")
            with open(src, "wb") as fh:
                fh.write(clip)
            _normalize_clip(src, dst)
            norm_paths.append(dst)

        if len(norm_paths) == 1:
            with open(norm_paths[0], "rb") as fh:
                return fh.read()

        list_path = os.path.join(d, "list.txt")
        with open(list_path, "w") as fh:
            for p in norm_paths:
                fh.write(f"file '{p}'\n")
        out = os.path.join(d, "merged.mp4")
        _run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
              "-c", "copy", "-movflags", "+faststart", out])
        with open(out, "rb") as fh:
            return fh.read()
