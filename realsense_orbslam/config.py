from dataclasses import dataclass


@dataclass
class Config:
    vocab: str = (
        "/workspaces/realsense_orbslam/third_party/orbslam3-python/ORB_SLAM3/Vocabulary/ORBvoc.txt"  # path to ORB-SLAM3 vocabulary file
    )
    settings: str = "d435_stereo.yaml"  # generated stereo settings file
    width: int = 640
    height: int = 480
    fps: int = 30
    emitter: bool = False  # IR dot projector off -> cleaner ORB features
    warmup_frames: int = 30  # let auto-exposure settle before tracking

    # Robustness
    reset_on_start: bool = True  # hardware-reset the camera before streaming
    frame_timeout_ms: int = 5000  # per-frame wait before counting a miss
    max_consecutive_misses: int = 10  # consecutive misses before auto-recovering
