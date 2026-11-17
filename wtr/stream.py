"""RTSP reader that always hands out the newest frame.

OpenCV buffers frames; if we process slower than the camera sends, a plain
read() loop falls further and further behind real time. A background thread
drains the stream and keeps only the latest frame.
"""

import os
import threading
import time

# Must be set before the first VideoCapture is opened. TCP avoids the smeared
# frames that RTSP-over-UDP produces on Wi-Fi.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402


class LatestFrameReader:
    def __init__(self, url: str, open_timeout_s: float = 10.0):
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(open_timeout_s * 1000),
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(open_timeout_s * 1000),
        ])
        if not self._cap.isOpened():
            raise ConnectionError("could not open stream")
        self._lock = threading.Lock()
        self._frame = None
        self._frame_time = 0.0
        self.frames_read = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + open_timeout_s
        while self._frame is None:
            if time.monotonic() > deadline:
                self.close()
                raise ConnectionError("stream opened but no frame arrived")
            time.sleep(0.05)

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._frame_time = time.time()
                self.frames_read += 1

    def read(self):
        """Return (frame, wall-clock time it was received)."""
        with self._lock:
            return self._frame, self._frame_time

    @property
    def codec(self) -> str:
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip() or "unknown"

    @property
    def declared_fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()
