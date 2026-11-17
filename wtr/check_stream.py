"""Check both Tapo streams: connect, report codec/resolution/fps, save a frame.

    uv run python -m wtr.check_stream
"""

import sys
import time

from wtr.config import ROOT, STREAMS, ConfigError, load_camera_config

MEASURE_SECONDS = 5


def check(cfg, stream: str) -> bool:
    import cv2

    from wtr.stream import LatestFrameReader

    print(f"\n[{stream}] {cfg.masked_url(stream)}")
    try:
        reader = LatestFrameReader(cfg.rtsp_url(stream))
    except Exception as exc:  # decoder errors can echo the URL: redact
        print(f"  FAILED: {cfg.redact(str(exc))}")
        print("  Check: camera IP, camera account user/password, Privacy Mode off.")
        return False

    start_count, start = reader.frames_read, time.monotonic()
    time.sleep(MEASURE_SECONDS)
    fps = (reader.frames_read - start_count) / (time.monotonic() - start)
    frame, _ = reader.read()
    height, width = frame.shape[:2]

    out = ROOT / "snapshots" / f"check_{stream}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), frame)

    print(f"  codec:      {reader.codec}")
    print(f"  resolution: {width}x{height}")
    print(f"  fps:        {fps:.1f} measured, {reader.declared_fps:.1f} declared")
    print(f"  snapshot:   {out.relative_to(ROOT)}")
    reader.close()
    return True


def main() -> int:
    try:
        cfg = load_camera_config()
    except ConfigError as exc:
        print(f"Config error: {exc}")
        return 2
    results = [check(cfg, stream) for stream in STREAMS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
