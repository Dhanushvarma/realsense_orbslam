import numpy as np
import viser

from pose import pose_components

# A new image-bearing breadcrumb frustum is dropped once the live pose has moved
# or rotated past either threshold; breadcrumbs beyond MAX_HISTORY are pruned to
# bound memory and bandwidth.
KEYFRAME_DIST = 0.05  # meters of translation, or
KEYFRAME_ANGLE = np.deg2rad(10.0)  # radians of rotation
IMAGE_DOWNSAMPLE = 4  # stride applied to breadcrumb images
MAX_HISTORY = 120  # max retained breadcrumbs (0 = unbounded)


def _rotation_angle(R_a, R_b):
    """Geodesic angle (radians) between two rotation matrices."""
    dR = R_a @ R_b.T
    return np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))


class SlamVisualizer:
    """Live web visualization (default http://<host>:8080). A bright frustum
    tracks the current pose and drops a trail of dimmer frustums, each showing
    the image captured at that pose."""

    def __init__(self, cfg, fx):
        self.server = viser.ViserServer(host="0.0.0.0", port=8080)
        self.server.scene.set_up_direction("-y")  # ORB-SLAM +y points down
        self.server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)

        # viser frustums use the OpenCV convention (+Z fwd, +X right, +Y down),
        # which matches ORB-SLAM3's camera frame -> Twc rotation drops straight in.
        self.fov = 2 * np.arctan2(cfg.height / 2, fx)
        self.aspect = cfg.width / cfg.height

        # Live "current pose" frustum (no image, just a bright marker).
        self.frustum = self.server.scene.add_camera_frustum(
            "/world/camera",
            fov=self.fov,
            aspect=self.aspect,
            scale=0.15,
            color=(255, 120, 0),
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
        )

        self.history = []  # breadcrumb handles, oldest first
        self._last_pos = None
        self._last_R = None
        self._count = 0

    def _is_new_keyframe(self, pos, R):
        if self._last_pos is None:
            return True
        moved = np.linalg.norm(pos - self._last_pos) > KEYFRAME_DIST
        turned = _rotation_angle(R, self._last_R) > KEYFRAME_ANGLE
        return moved or turned

    def update(self, Twc, image=None):
        pos, quat_wxyz, _ = pose_components(Twc)

        # Move the live frustum (updating handle attrs avoids re-adding the node).
        self.frustum.position = pos
        self.frustum.wxyz = quat_wxyz

        R = Twc[:3, :3]
        if image is None or not self._is_new_keyframe(pos, R):
            return

        # Downsample + ensure 3 channels (IR frames are single-channel grayscale).
        img = image[::IMAGE_DOWNSAMPLE, ::IMAGE_DOWNSAMPLE]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        handle = self.server.scene.add_camera_frustum(
            f"/world/history/f{self._count}",
            fov=self.fov,
            aspect=self.aspect,
            scale=0.08,
            color=(80, 160, 255),
            wxyz=quat_wxyz,
            position=pos,
            image=img,
        )
        self.history.append(handle)
        self._count += 1
        self._last_pos, self._last_R = pos, R

        # Drop the oldest breadcrumb once we exceed the cap.
        if MAX_HISTORY and len(self.history) > MAX_HISTORY:
            self.history.pop(0).remove()
