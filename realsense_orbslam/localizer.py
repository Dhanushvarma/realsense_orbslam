"""Stereo ORB-SLAM3 wrapped as an episodic localizer for closed-loop control.

The intended use is an imitation-learning deployment loop: at the start of an
episode you call ``reset()`` to declare "here is the world/odom origin", and from
then on ``get_pose()`` returns the camera's pose *relative to that origin* in
metres. A background thread keeps SLAM fed from the RealSense at full frame rate,
so ``get_pose()`` is a cheap, non-blocking read of the most recent estimate.

    with StereoLocalizer() as loc:
        loc.reset()                      # episode start -> defines odom origin
        while running:
            sample = loc.get_pose()      # non-blocking, returns latest estimate
            if sample is not None and sample.tracking_ok:
                obs = sample.position    # (x, y, z) in metres, relative to reset

Frame conventions
-----------------
ORB-SLAM3 reports the camera *optical* frame: +Z forward (out of the lens),
+X right, +Y down. If your policy expects a robot/ROS "FLU" frame (+X forward,
+Y left, +Z up) pass ``T_body_cam=make_T(OPTICAL_TO_FLU)`` (or your real
body<-camera extrinsic). The origin is re-anchored at every ``reset()``, so the
odom frame's axes are whatever the body frame's axes are at reset time.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import orbslam3

from calibration import export_settings
from camera import RealSenseStereo, reset_device
from config import Config
from pose import pose_components

# Rotation mapping the camera optical frame (X right, Y down, Z forward) to a
# robot FLU body frame (X forward, Y left, Z up). Columns are the optical basis
# vectors expressed in the body frame. Use as the rotation block of T_body_cam
# when the camera looks straight forward; add your own translation lever-arm.
OPTICAL_TO_FLU = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def make_T(R=np.eye(3), t=(0.0, 0.0, 0.0)):
    """Assemble a 4x4 homogeneous transform from a rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class PoseSample:
    """A single localization estimate, relative to the current odom origin."""

    T: np.ndarray  # 4x4 body-in-odom (identity at the moment of reset)
    position: np.ndarray  # (3,) metres in the odom frame
    quat_wxyz: np.ndarray  # (4,) orientation, w-first
    tracking_ok: bool  # False -> SLAM lost/initialising; pose is stale
    stamp: float  # camera timestamp of this estimate (seconds)
    frame_index: int  # monotonic counter of processed frames


class StereoLocalizer:
    """Background-threaded stereo ORB-SLAM3 localizer with an episodic origin.

    Owns the RealSense camera and the SLAM system. ``start()`` spins up a worker
    thread that streams frames into SLAM continuously; the foreground (your
    control loop) only ever calls the cheap, lock-guarded ``get_pose()``.
    """

    def __init__(self, cfg=None, T_body_cam=None, visualize=False):
        self.cfg = cfg or Config()
        # body <- camera-optical extrinsic. Identity => report the optical frame.
        self.T_body_cam = np.eye(4) if T_body_cam is None else np.asarray(T_body_cam)
        self.visualize = visualize

        self._slam = None
        self._cam = None
        self._viz = None

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # Shared state, guarded by _lock.
        self._T_world_body = None  # latest body-in-(slam world), 4x4 or None
        self._T_origin_world = None  # inv(T_world_body at reset); None => unanchored
        self._tracking_ok = False
        self._stamp = 0.0
        self._frame_index = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Load SLAM, open the camera, and start the background tracking thread.

        Blocks until the camera is streaming (vocabulary load is the slow part).
        """
        if self.cfg.reset_on_start:
            reset_device()

        fx = export_settings(self.cfg)

        print("loading SLAM (vocabulary)...")
        self._slam = orbslam3.system(
            self.cfg.vocab, self.cfg.settings, orbslam3.Sensor.STEREO
        )
        self._slam.initialize()

        if self.visualize:
            from visualizer import SlamVisualizer  # optional dependency (viser)

            self._viz = SlamVisualizer(self.cfg, fx)

        self._cam = RealSenseStereo(self.cfg)
        self._cam.start()

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="orbslam", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the worker thread and release the camera and SLAM system."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._slam is not None:
            self._slam.shutdown()
            self._slam = None
        if self._cam is not None:
            self._cam.stop()
            self._cam = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- episodic origin ---------------------------------------------------

    def reset(self, wait=True, timeout=10.0):
        """Anchor the odom origin at the current pose (call this at episode start).

        With ``wait=True`` (default) blocks until tracking is OK before anchoring,
        so the origin is taken from a valid pose. Returns True if anchored, False
        if it timed out without a tracked pose.

        Note: this re-anchors the *reported* frame only; it does not wipe the
        SLAM map (so relocalization back to earlier scenery still works). Use
        ``hard_reset()`` if you want SLAM to forget the map entirely.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._T_world_body is not None and (
                    self._tracking_ok or not wait
                ):
                    self._T_origin_world = np.linalg.inv(self._T_world_body)
                    return True
            if not wait or time.time() > deadline:
                return False
            time.sleep(0.01)

    def hard_reset(self, wait=True, timeout=10.0):
        """Wipe the SLAM map and re-anchor from scratch (full episode reset)."""
        with self._lock:
            self._T_world_body = None
            self._T_origin_world = None
            self._tracking_ok = False
        self._slam.reset()
        return self.reset(wait=wait, timeout=timeout)

    # -- pose access -------------------------------------------------------

    def get_pose(self):
        """Return the latest pose relative to the odom origin, or None.

        Non-blocking. Returns None if no pose has ever been estimated. If the
        origin has not been set yet it is lazily anchored to the first pose, so
        the result is always relative to *some* well-defined origin; call
        ``reset()`` at episode start to control exactly where that is.

        Check ``sample.tracking_ok`` -- when SLAM is lost the returned pose is the
        last good one (stale), not a fresh estimate.
        """
        with self._lock:
            if self._T_world_body is None:
                return None
            if self._T_origin_world is None:
                self._T_origin_world = np.linalg.inv(self._T_world_body)
            T = self._T_origin_world @ self._T_world_body
            tracking_ok = self._tracking_ok
            stamp = self._stamp
            frame_index = self._frame_index

        pos, quat_wxyz, _ = pose_components(T)
        return PoseSample(
            T=T,
            position=pos,
            quat_wxyz=quat_wxyz,
            tracking_ok=tracking_ok,
            stamp=stamp,
            frame_index=frame_index,
        )

    def wait_until_tracking(self, timeout=30.0):
        """Block until SLAM reports a tracked pose; return True if it did."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._tracking_ok:
                    return True
            time.sleep(0.02)
        return False

    @property
    def tracking_ok(self):
        with self._lock:
            return self._tracking_ok

    # -- worker ------------------------------------------------------------

    def _loop(self):
        """Background thread: pull frames, step SLAM, publish the latest pose."""
        misses = 0
        while not self._stop.is_set():
            frame = self._cam.read()
            if frame is None:
                misses += 1
                if misses >= self.cfg.max_consecutive_misses:
                    self._cam.recover()
                    misses = 0
                continue
            misses = 0

            left, right, stamp = frame
            self._slam.process_image_stereo(left, right, stamp)

            ok = self._slam.get_tracking_state() == orbslam3.TrackingState.OK
            if ok:
                T_world_cam = np.linalg.inv(np.asarray(self._slam.get_current_pose()))
                T_world_body = T_world_cam @ np.linalg.inv(self.T_body_cam)
                with self._lock:
                    self._T_world_body = T_world_body
                    self._tracking_ok = True
                    self._stamp = stamp
                    self._frame_index += 1
                if self._viz is not None:
                    self._viz.update(T_world_cam, image=left)
            else:
                with self._lock:
                    self._tracking_ok = False  # keep last T_world_body as stale


if __name__ == "__main__":
    # Smoke test: print pose relative to the reset origin at ~10 Hz.
    with StereoLocalizer(visualize=False) as loc:
        print("waiting for tracking to initialize...")
        if not loc.wait_until_tracking(timeout=30.0):
            raise SystemExit("SLAM never initialized -- check lighting/texture")
        loc.reset()
        print("origin anchored; moving the camera should change the pose")
        try:
            while True:
                s = loc.get_pose()
                if s is not None:
                    p, flag = s.position, "OK " if s.tracking_ok else "LOST"
                    print(f"[{flag}] x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
