import time

import numpy as np
import pyrealsense2 as rs

# A single shared context for the whole process. Creating and destroying multiple
# rs.context() objects in overlapping lifetimes is a known cause of
# "pure virtual method called" crashes at garbage-collection time, so we keep
# exactly one context alive and route everything (queries, reset, pipeline)
# through it.
CTX = rs.context()


def wait_for_device(timeout=20.0):
    """Block until a RealSense is present AND its IR profiles are queryable.

    After a hardware reset the device re-appears in the list before its sensors
    are fully constructed; probing the profiles is what tells us it's actually
    usable (not just enumerated)."""
    deadline = time.time() + timeout
    while True:
        devs = CTX.query_devices()
        if len(devs) > 0:
            try:
                profs = devs[0].first_depth_sensor().get_stream_profiles()
                if any(p.stream_type() == rs.stream.infrared for p in profs):
                    return devs[0]
            except RuntimeError:
                pass  # sensors still coming up after reset
        if time.time() > deadline:
            raise RuntimeError("RealSense device did not become usable")
        time.sleep(0.5)


def reset_device(settle=5.0):
    """Power-cycle the camera over USB to clear a wedged state from a prior run,
    then block until it has fully re-enumerated."""
    devs = CTX.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense device connected")
    print("hardware reset...")
    devs[0].hardware_reset()
    time.sleep(settle)        # device drops off the bus and starts re-enumerating
    wait_for_device()         # ...block until it's back and its profiles are ready
    print("device ready")


class RealSenseStereo:
    """Streams the two infrared imagers (the D435's hardware-rectified stereo pair)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None

    def _config(self):
        rscfg = rs.config()
        c = self.cfg
        rscfg.enable_stream(rs.stream.infrared, 1, c.width, c.height, rs.format.y8, c.fps)
        rscfg.enable_stream(rs.stream.infrared, 2, c.width, c.height, rs.format.y8, c.fps)
        return rscfg

    def start(self):
        wait_for_device()
        self.pipeline = rs.pipeline(CTX)  # share the one context
        profile = self.pipeline.start(self._config())
        sensor = profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1 if self.cfg.emitter else 0)
        for _ in range(self.cfg.warmup_frames):
            self.pipeline.wait_for_frames()

    def read(self, timeout_ms=None):
        """Return (left, right, timestamp_seconds), or None on timeout / missing frame."""
        timeout_ms = timeout_ms or self.cfg.frame_timeout_ms
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None
        left, right = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
        if not left or not right:
            return None
        return (
            np.asanyarray(left.get_data()),
            np.asanyarray(right.get_data()),
            left.get_timestamp() * 1e-3,  # ms -> s
        )

    def recover(self):
        """Stop, hardware-reset, restart -- for use after repeated timeouts."""
        self.stop()
        reset_device()
        self.start()

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass  # already stopped / device gone
            self.pipeline = None