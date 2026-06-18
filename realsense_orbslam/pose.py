import numpy as np
from scipy.spatial.transform import Rotation


def pose_components(Twc):
    """Decompose a camera-in-world pose (4x4) into useful parts.

    Returns:
        position   : (3,) translation in world frame
        quat_wxyz  : (4,) quaternion, w-first (viser convention)
        euler_deg  : (3,) intrinsic XYZ Euler angles in degrees
    """
    rot = Rotation.from_matrix(Twc[:3, :3])
    quat_wxyz = np.roll(rot.as_quat(), 1)  # scipy returns xyzw -> reorder to wxyz
    return Twc[:3, 3], quat_wxyz, rot.as_euler("xyz", degrees=True)
