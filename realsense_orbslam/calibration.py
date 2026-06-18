import pyrealsense2 as rs

from camera import wait_for_device


def _write_yaml(path, cfg, K1, K2, baseline):
    with open(path, "w") as f:
        f.write(f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: {K1.fx}
Camera1.fy: {K1.fy}
Camera1.cx: {K1.ppx}
Camera1.cy: {K1.ppy}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0

Camera2.fx: {K2.fx}
Camera2.fy: {K2.fy}
Camera2.cx: {K2.ppx}
Camera2.cy: {K2.ppy}
Camera2.k1: 0.0
Camera2.k2: 0.0
Camera2.p1: 0.0
Camera2.p2: 0.0

Camera.width: {cfg.width}
Camera.height: {cfg.height}
Camera.fps: {cfg.fps}
Camera.RGB: 1

Stereo.ThDepth: 40.0
Stereo.b: {baseline:.6f}
Stereo.T_c1_c2: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [1.0, 0.0, 0.0, {baseline:.6f}, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

ORBextractor.nFeatures: 1200
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
""")


def _read_calibration(cfg):
    """Read IR intrinsics + baseline without starting a pipeline.

    Every read happens here, while device/sensor/profiles are still in scope:
    pyrealsense2 profiles don't keep their parent device alive, so reading a
    returned profile later hits freed C++ objects (-> "pure virtual method
    called"). The rs.intrinsics / rs.extrinsics value structs are safe to return.
    """
    dev = wait_for_device()
    sensor = dev.first_depth_sensor()  # Stereo Module exposes the IR profiles

    ir1 = ir2 = None
    for p in sensor.get_stream_profiles():
        if p.stream_type() != rs.stream.infrared or not p.is_video_stream_profile():
            continue
        v = p.as_video_stream_profile()
        if (v.width(), v.height(), v.fps()) != (cfg.width, cfg.height, cfg.fps):
            continue
        if p.stream_index() == 1:
            ir1 = v
        elif p.stream_index() == 2:
            ir2 = v

    if ir1 is None or ir2 is None:
        raise RuntimeError(
            f"No IR profile for {cfg.width}x{cfg.height}@{cfg.fps}. "
            "Run `rs-enumerate-devices` to list supported modes."
        )

    # Read everything now, while dev/sensor/ir1/ir2 are alive.
    K1 = ir1.get_intrinsics()
    K2 = ir2.get_intrinsics()
    baseline = abs(ir1.get_extrinsics_to(ir2).translation[0])  # meters
    return K1, K2, float(baseline)


def export_settings(cfg):
    """Read calibration, write the stereo YAML, return left-camera fx for the FoV."""
    K1, K2, baseline = _read_calibration(cfg)
    _write_yaml(cfg.settings, cfg, K1, K2, baseline)
    print(f"settings: fx={K1.fx:.1f} cx={K1.ppx:.1f} baseline={baseline * 1000:.1f}mm")
    return K1.fx
