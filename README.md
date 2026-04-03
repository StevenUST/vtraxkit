# Vtraxkit

Extract multi-person skeleton trajectories from videos with one line of code.

```python
import vtraxkit

tracks = vtraxkit.extract("video.mp4")
tracks.filter(min_duration=2.0).save("output.npy")
```

## How It Works

```
Video → Person Detection (YOLO) → Multi-Person Tracking (ByteTrack) → Pose Estimation (MediaPipe) → Skeleton Trajectories
```

Pose estimation runs **after** tracking and filtering, so compute is only spent on valid tracks.

## Installation

```bash
pip install vtraxkit
pip install vtraxkit[yolo,mediapipe]  # with detection + pose backends
```

### Optional dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `yolo` | ultralytics | Person detection + tracking |
| `mediapipe` | mediapipe | Pose estimation |
| `pandas` | pandas | DataFrame export |
| `full` | all of the above | Everything |

## Quick Start

```python
import vtraxkit

# Extract with default settings
tracks = vtraxkit.extract("video.mp4")

# Extract with options
tracks = vtraxkit.extract(
    "video.mp4",
    device="cuda",
    detector="yolo:yolov8s.pt",
    frame_skip=3,
    min_duration=1.0,
)

# Filter + save
tracks.filter(min_duration=2.0, min_frames=10).save("output.npy")

# Access skeleton data
for track in tracks:
    pose = track.skeleton_array("pose")  # shape: (T, 33, 4)
    print(f"Track {track.track_id}: {track.duration:.1f}s, {track.num_frames} frames")

# Load saved tracks
tracks = vtraxkit.load("output.npy")

# Export to DataFrame
df = tracks.to_dataframe()
```

## Reusable Pipeline

For processing multiple videos, create a `Pipeline` to avoid reloading models:

```python
from vtraxkit import Pipeline

pipeline = Pipeline(device="cuda")
for video in video_list:
    tracks = pipeline.run(video)
    tracks.save(f"{video}.npy")
pipeline.close()
```

## Output Format

Each track contains:

| Field | Type | Description |
|-------|------|-------------|
| `track_id` | int | Unique person ID |
| `frames` | list[int] | Frame indices |
| `timestamps` | list[float] | Timestamps (seconds) |
| `bboxes` | list[BBox] | Bounding boxes (x, y, w, h) |
| `skeletons` | list[Skeleton] | Keypoints per frame |

Skeleton keypoint groups (via MediaPipe Holistic):

| Group | Keypoints | Dimensions |
|-------|-----------|------------|
| `pose` | 33 | x, y, z, visibility |
| `left_hand` | 21 | x, y, z, visibility |
| `right_hand` | 21 | x, y, z, visibility |
| `face` | 468 | x, y, z |

## Requirements

- Python >= 3.9
- numpy >= 1.20
- opencv-python >= 4.5

## License

This project is licensed under the [MIT License](LICENSE).

Note: The optional YOLO detection backend depends on
[ultralytics](https://github.com/ultralytics/ultralytics), which is licensed
under AGPL-3.0. If you use `vtraxkit[yolo]`, please ensure your usage
complies with the AGPL-3.0 terms. For AGPL-free usage, you may implement a
custom detection backend without the ultralytics dependency.
