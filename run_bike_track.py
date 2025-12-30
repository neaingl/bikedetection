"""Bike detection and tracking on AVI videos using YOLO + ByteTrack."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import torch
from ultralytics import YOLO


BICYCLE_CLASS_ID = 1
CODEC_PREFERENCE = ["mp4v", "XVID", "MJPG"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track bicycles in an AVI video using YOLO + ByteTrack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(Path("input") / "intersection.avi"),
        help="Path to input AVI video.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8m.pt",
        help="YOLO weights to use (e.g., yolov8s.pt, yolov8m.pt, yolov8l.pt).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: 'auto', 'cpu', or CUDA device index (e.g., 0).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference.")
    parser.add_argument(
        "--tracker",
        type=str,
        default=str(Path("bytetrack_custom.yaml")),
        help="Path to ByteTrack tracker configuration.",
    )
    return parser.parse_args()


def select_device(arg_device: str) -> str | int:
    if arg_device.lower() == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if arg_device.lower() == "cpu":
        return "cpu"
    return arg_device


def get_video_properties(video_path: Path) -> Tuple[int, int, float, int | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count if frame_count > 0 else None


def init_writer(out_path: Path, width: int, height: int, fps: float) -> Tuple[cv2.VideoWriter, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for codec in CODEC_PREFERENCE:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec {codec} for output.")
            return writer, codec
    raise RuntimeError("Failed to initialize VideoWriter with preferred codecs (mp4v, XVID, MJPG).")


def filter_bicycle_boxes(result) -> Iterable[Tuple[Tuple[int, int, int, int], float, int | None]]:
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return []

    classes = boxes.cls.int().cpu().numpy()
    mask = classes == BICYCLE_CLASS_ID
    if not mask.any():
        return []

    xyxy = boxes.xyxy[mask].cpu().numpy()
    confs = boxes.conf[mask].cpu().numpy()
    ids = boxes.id[mask].cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

    filtered = []
    for coords, conf, track_id in zip(xyxy, confs, ids):
        x1, y1, x2, y2 = map(int, coords.tolist())
        filtered.append(((x1, y1, x2, y2), float(conf), int(track_id) if track_id is not None else None))
    return filtered


def draw_detections(frame, detections: Iterable[Tuple[Tuple[int, int, int, int], float, int | None]]):
    for (x1, y1, x2, y2), conf, track_id in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"bicycle {conf:.2f}"
        if track_id is not None:
            label += f" id:{track_id}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def process_video(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    tracker_path = Path(args.tracker)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}. Place the .avi in ./input or pass --video.")
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker configuration not found: {tracker_path}")

    device = select_device(args.device)
    width, height, fps, frame_count = get_video_properties(video_path)
    print(
        f"Starting bike detection & tracking\n"
        f"Video: {video_path}\nWeights: {args.weights}\nDevice: {device}\n"
        f"Confidence: {args.conf}, IoU: {args.iou}, ImgSz: {args.imgsz}\n"
        f"Tracker: {tracker_path}"
    )

    model = YOLO(args.weights)
    writer, codec_used = init_writer(Path("output") / "out_bike_track.avi", width, height, fps)

    total_frames = 0
    total_bike_detections = 0
    max_bikes_single_frame = 0

    results_generator = model.track(
        source=str(video_path),
        tracker=str(tracker_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        classes=[BICYCLE_CLASS_ID],
        stream=True,
        persist=True,
        verbose=False,
    )

    try:
        for total_frames, result in enumerate(results_generator, start=1):
            frame = result.orig_img
            if frame is None:
                continue

            frame_to_draw = frame.copy()
            detections = list(filter_bicycle_boxes(result))
            draw_detections(frame_to_draw, detections)

            bike_count = len(detections)
            total_bike_detections += bike_count
            max_bikes_single_frame = max(max_bikes_single_frame, bike_count)

            writer.write(frame_to_draw)

            if frame_count:
                print(
                    f"Processed frame {total_frames}/{frame_count} | bikes this frame: {bike_count} | max so far: {max_bikes_single_frame}",
                    end="\r",
                )
            else:
                print(
                    f"Processed frame {total_frames} | bikes this frame: {bike_count} | max so far: {max_bikes_single_frame}",
                    end="\r",
                )
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"\nError during processing: {exc}", file=sys.stderr)
        return 1
    finally:
        writer.release()

    average_bikes = total_bike_detections / total_frames if total_frames else 0.0
    print(
        f"\nFinished processing.\n"
        f"Total frames: {total_frames}\n"
        f"Max bicycles in a single frame: {max_bikes_single_frame}\n"
        f"Average bicycles per frame: {average_bikes:.2f}\n"
        f"Output saved to: output/out_bike_track.avi (codec: {codec_used})"
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return process_video(args)
    except FileNotFoundError as err:
        print(f"[File error] {err}", file=sys.stderr)
        return 1
    except Exception as err:  # pragma: no cover - runtime guard
        print(f"[Unexpected error] {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
