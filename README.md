# Bicycle Detection & Tracking (AVI) with YOLO + ByteTrack

Detect and track bicycles in AVI videos using Ultralytics YOLO and ByteTrack. The script downloads YOLO weights automatically, runs detection + tracking, and exports an annotated AVI with stable track IDs.

## Project Layout
```
.
├─ input/                 # Place your input AVI video here (default: intersection.avi)
├─ output/                # Generated outputs (out_bike_track.avi)
├─ run_bike_track.py      # Main entrypoint
├─ bytetrack_custom.yaml  # Tracker configuration you can tune
└─ requirements.txt       # Minimal dependencies
```

## Installation
Example with a virtual environment:

```bash
# From the repository root
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Run the script (defaults use `input/intersection.avi`, YOLOv8m weights, and auto device selection):

```bash
python run_bike_track.py
```

Common options:

```bash
python run_bike_track.py \
  --video input/your_video.avi \
  --weights yolov8s.pt \
  --device 0 \
  --conf 0.3 \
  --iou 0.5 \
  --imgsz 720 \
  --tracker bytetrack_custom.yaml
```

Notes:
- Use `--device 0` (or other CUDA index) to force GPU, `--device cpu` to force CPU. Default is auto-detect.
- Only the COCO `bicycle` class (ID 1) is kept. Boxes are green with confidence and the ByteTrack `id` label.
- Output video: `output/out_bike_track.avi` keeps the same resolution and FPS when available. The writer prefers `mp4v`/`XVID` codecs and falls back to `MJPG`.
- If your input video is missing, place an `.avi` file in `input/` or pass `--video path/to/file.avi`.

## Tracker tuning (ByteTrack)
Adjust `bytetrack_custom.yaml` to balance ID stability vs. sensitivity:

- `track_thresh`: Raise to reduce false positives; lower to detect smaller bikes.
- `match_thresh`: Raise (e.g., 0.85) to reduce ID switches; lower if tracks are frequently lost.
- `track_buffer`: Increase to keep IDs during short occlusions; decrease to drop stale tracks faster.
- `min_box_area`: Increase to ignore tiny boxes/noise.
- `frame_rate`: Set to your video FPS for more consistent behavior.

After editing the YAML, pass it via `--tracker` (default already points to `bytetrack_custom.yaml`).

## Output & Stats
During processing, the script prints progress, the maximum simultaneous bicycle count, and the average bicycles per frame. On success, it returns 0 and saves the annotated video to `output/out_bike_track.avi`.

## Troubleshooting
- **FFmpeg/codecs**: If `mp4v`/`XVID` codecs are unavailable, the script falls back to `MJPG`. Install/enable relevant codecs (e.g., `sudo apt-get install libxvidcore-dev`) if AVI writing fails.
- **CUDA not used**: Ensure the correct NVIDIA drivers are installed and pass `--device 0`. If CUDA is unavailable, the script automatically runs on CPU.
- **Model download slow/fails**: Provide a local weights file via `--weights /path/to/yolov8m.pt`.
- **No bikes detected**: Lower `--conf` or `track_thresh` in `bytetrack_custom.yaml`, or increase `--imgsz` for higher resolution inference.
