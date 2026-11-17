"""Time candidate models on this machine's CPU.

Run it on the machine that will do the real work (the Umbrel PC): timings
from a different CPU say nothing about that one.

    uv run python -m wtr.benchmark                      # all models, ONNX Runtime
    uv run python -m wtr.benchmark --backend openvino   # Intel CPUs
    uv run python -m wtr.benchmark --image snapshots/check_hi.jpg
"""

import argparse
import platform
import statistics
import time

import cv2

YOLO_MODELS = {
    "yolo26n (people + remote)": "yolo26n.pt",
    "yolo26n-pose (wrists, whole frame)": "yolo26n-pose.pt",
}
EXPORT_FORMAT = {"onnx": "onnx", "openvino": "openvino"}
RTMLIB_BACKEND = {"onnx": "onnxruntime", "openvino": "openvino"}


def time_call(fn, runs: int, warmup: int = 3) -> tuple[float, float]:
    """Median and worst wall-clock milliseconds of fn()."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), max(samples)


def report(name: str, median_ms: float, worst_ms: float, note: str = ""):
    print(f"  {name:<38} {median_ms:7.1f} ms  {1000 / median_ms:6.1f} fps  "
          f"(worst {worst_ms:.0f} ms) {note}")


def bench_yolo(image, backend: str, imgsz: int, runs: int):
    from ultralytics import YOLO

    for name, weights in YOLO_MODELS.items():
        exported = YOLO(weights).export(format=EXPORT_FORMAT[backend], imgsz=imgsz, verbose=False)
        model = YOLO(exported, task="pose" if "pose" in weights else "detect")
        result = model(image, imgsz=imgsz, device="cpu", verbose=False)[0]
        median, worst = time_call(
            lambda: model(image, imgsz=imgsz, device="cpu", verbose=False), runs)
        report(name, median, worst, f"[{len(result.boxes)} detections @ {imgsz}px]")


def bench_rtmpose(image, backend: str, runs: int):
    from rtmlib import Body

    body = Body(mode="lightweight", backend=RTMLIB_BACKEND[backend], device="cpu")
    boxes = body.det_model(image)
    if len(boxes) == 0:
        print("  rtmpose: no person in the test image, skipped")
        return
    median, worst = time_call(lambda: body.det_model(image), runs)
    # rtmlib's "lightweight" preset is YOLOX-tiny + RTMPose-s (not RTMPose-t).
    report("yolox-tiny (people)", median, worst, f"[{len(boxes)} people]")
    one_box = [boxes[0]]
    median, worst = time_call(lambda: body.pose_model(image, bboxes=one_box), runs)
    report("rtmpose-s (wrists, per person)", median, worst)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["onnx", "openvino"], default="onnx")
    parser.add_argument("--image", help="test image (default: a sample photo with people)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    if args.image:
        path = args.image
    else:
        from ultralytics.utils import ASSETS
        path = str(ASSETS / "bus.jpg")
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")

    print(f"CPU: {platform.processor() or platform.machine()}  backend: {args.backend}  "
          f"image: {image.shape[1]}x{image.shape[0]}  runs: {args.runs}")
    bench_yolo(image, args.backend, args.imgsz, args.runs)
    bench_rtmpose(image, args.backend, args.runs)


if __name__ == "__main__":
    main()
