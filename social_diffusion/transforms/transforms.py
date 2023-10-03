import numpy as np
import torch.nn.functional as F
from einops import rearrange
import torch
import numpy.linalg as la
import social_diffusion.transforms.rotation as rot
from social_diffusion import WeirdShapeError


class NormalizeDoesNotAllowZeroZError(ValueError):
    def __init__(self, msg):
        super().__init__(msg)


def apply_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    is_flat = False
    if len(seq.shape) == 2:
        is_flat = True
        seq = rearrange(seq, "t (j d) -> t j d", d=3)

    if len(seq.shape) != 3 or seq.shape[2] != 3:
        raise WeirdShapeError(f"weird shape: {seq.shape}")

    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    seq = seq - mu
    seq_rot = rot.apply_rotation_to_seq(seq, R)

    if is_flat:
        seq_rot = rearrange(seq, "t j d -> t (j d)")
    return seq_rot


def get_normalization(left3d, right3d):
    """
    Get rotation + translation to center and face along the x-axis
    """
    mu = (left3d + right3d) / 2
    mu[2] = 0
    left2d = left3d[:2]
    right2d = right3d[:2]
    y = right2d - left2d
    y = y / (la.norm(y) + 0.00000001)
    angle = np.arctan2(y[1], y[0])
    R = rot.rot3d(0, 0, angle)
    return mu, R


def undo_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    is_flat = False
    if len(seq.shape) == 2:
        is_flat = True
        seq = rearrange(seq, "t (j d) -> t j d", d=3)
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    R_T = np.transpose(R)
    seq = rot.apply_rotation_to_seq(seq, R_T)
    seq = seq + mu
    if is_flat:
        seq = rearrange(seq, "t j d -> t (j d)")
    return seq


def tn_get_normalization(left3d, right3d, allow_zero_z=False):
    """
    :param left3d: {n_batch x 3}
    :param right3d: {n_batch x 3}
    Get rotation + translation to center and face along the x-axis
    """
    if not allow_zero_z:
        eps = 0.00000001
        left_z = eps > torch.abs(left3d[:, 2])
        right_z = eps > torch.abs(right3d[:, 2])
        if torch.count_nonzero(left_z) > 0 or torch.count_nonzero(right_z) > 0:
            print(
                f"left-> nonzero:{torch.count_nonzero(left_z)}, z:{left_z.shape}"  # noqa E501
            )  # noqa E501
            if torch.count_nonzero(left_z) > 0:
                for idx in torch.nonzero(left_z):
                    print(f"\t{idx} => {torch.round(left3d[idx], decimals=2)}")
                    if idx - 1 >= 0:
                        print(
                            f"\t\t t-1 => {torch.round(left3d[idx-1], decimals=2)}"  # noqa E501
                        )  # noqa E501
                    if idx + 1 < len(left3d):
                        print(
                            f"\t\t t+1 => {torch.round(left3d[idx+1], decimals=2)}"  # noqa E501
                        )  # noqa E501

            print(
                f"right-> nonzero:{torch.count_nonzero(right_z)}, z:{right_z.shape}"  # noqa E501
            )  # noqa E501
            if torch.count_nonzero(right_z) > 0:
                for idx in torch.nonzero(right_z):
                    print(f"\t{idx} => {right3d[idx]}")
            raise NormalizeDoesNotAllowZeroZError("We do not allow z:0!")

    mu = (left3d + right3d) / 2
    mu[:, 2] = 0

    left2d = left3d[:, :2]
    right2d = right3d[:, :2]
    y = right2d - left2d
    y = y / (torch.linalg.norm(y, dim=1, keepdims=True) + 0.0000001)
    a = y[:, 0]
    b = y[:, 1]
    angle = torch.atan2(b, a)

    R = rot.tn_rot3d_c(angle)
    return R, mu


def tn_batch_rotpoint_to_rvec(rotpoint, device=None):
    """ """
    is_numpy = False
    if isinstance(rotpoint, np.ndarray):
        is_numpy = True
        rotpoint = torch.from_numpy(rotpoint)
    if device is None:
        device = torch.device("cpu")
    if len(rotpoint.shape) != 2 or rotpoint.shape[1] != 3:
        raise ValueError(f"Weird rvec shape: {rotpoint.shape}")
    rotpoint = rotpoint.to(device)
    rotpoint = rotpoint / torch.norm(rotpoint, dim=1, keepdim=True)
    a = rotpoint[:, 0]
    b = rotpoint[:, 1]
    angle = torch.atan2(b, a)
    R = rot.tn_rot3d_c(angle)
    rvec = matrix_to_axis_angle(R)
    if is_numpy:
        rvec = rvec.cpu().numpy()
    return rvec


def tn_batch_rvec_to_rotpoint(rvec, device=None):
    """
    converts the rotation vector into a 3d point (that lives on the xy plane)
    The 3d point does not exhibit discontinuities!
    :param rvec: {n_batch x 3}
    """
    is_numpy = False
    if isinstance(rvec, np.ndarray):
        is_numpy = True
        rvec = torch.from_numpy(rvec)
    if device is None:
        device = torch.device("cpu")

    if len(rvec.shape) != 2 or rvec.shape[1] != 3:
        raise ValueError(f"Weird rvec shape: {rvec.shape}")
    rvec = rvec.to(device)
    Rs = axis_angle_to_matrix(rvec)
    pos2d = torch.zeros_like(rvec)
    pos2d[:, 0] = 1
    rotpoint = torch.bmm(Rs, pos2d.unsqueeze(2)).squeeze(2)
    if is_numpy:
        rotpoint = rotpoint.cpu().numpy()
    return rotpoint


def tn_batch_normalize(
    Pose: np.ndarray,
    skel,
    *,
    return_transform=False,
    allow_zero_z=False,
    device=None,  # noqa E501
):
    """
    :param Pose: {n_batch x jd}
    """
    Pose = skel.fix_shape(Pose, unroll_jd=True)
    if device is None:
        device = torch.device("cpu")
    if isinstance(Pose, np.ndarray):
        Pose = torch.from_numpy(Pose)
    if not torch.is_tensor(Pose):
        raise ValueError("<Pose> must be a torch Tensor!")

    left_jid = skel.normalizing_left_jid
    right_jid = skel.normalizing_right_jid

    Pose = Pose.to(device)

    Left3d = Pose[:, left_jid]
    Right3d = Pose[:, right_jid]

    from einops import reduce

    sum_left = (
        reduce(np.abs(Left3d.cpu().numpy()), "t d -> t", "sum") < 0.0001
    ) * 1  # noqa E501
    sum_right = (
        reduce(np.abs(Right3d.cpu().numpy()), "t d -> t", "sum") < 0.0001
    ) * 1  # noqa E501
    if np.count_nonzero(sum_left) > 0:
        print("left!!", np.count_nonzero(sum_left), sum_left.shape)
        print("right!!", np.count_nonzero(sum_right))
        exit()
    if np.count_nonzero(sum_right) > 0:
        print("right!!", np.count_nonzero(sum_right))
        exit()

    Rs, Mus = tn_get_normalization(
        left3d=Left3d, right3d=Right3d, allow_zero_z=allow_zero_z
    )
    Pose = tn_framewise_transform(poses=Pose, mu=Mus, R=Rs)

    if return_transform:
        return Pose, Rs, Mus
    else:
        return Pose


def tn_framewise_transform(poses, mu, R):
    """
    :param poses: {n_batch x j x 3}
    :param mu: {n_batch x 3}
    :param R: {n_batch x 3x3}
    """
    n_batch = poses.size(0)
    is_flat = False
    if len(poses.shape) == 2:
        is_flat = True
        poses = poses.reshape(n_batch, -1, 3)
    mu = mu.unsqueeze(1)
    poses = poses - mu
    output_poses = poses @ R
    if is_flat:
        output_poses = output_poses.reshape((n_batch, -1))
    return output_poses


def tn_framewise_undo_transform(poses, mu, R):
    """
    :param poses: {n_batch x j x 3}
    :param mu: {n_batch x 3}
    :param R: {n_batch x 3x3}
    """
    n_batch = poses.size(0)
    is_flat = False
    if len(poses.shape) == 2:
        is_flat = True
        poses = poses.reshape(n_batch, -1, 3)
    R_T = torch.transpose(R, 1, 2)
    mu = mu.unsqueeze(1)
    poses = poses @ R_T
    output_poses = poses + mu
    if is_flat:
        output_poses = output_poses.reshape((n_batch, -1))
    return output_poses


# ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~
# --- we do not want to have dependency on pytorch3d! ---
# instead we just graft the functions from their repo!
# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
# ALL CODE BELOW IS FROM PYTORCH3D!
# ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~ ~~


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and
    # `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,  # noqa E501
    )
    return quaternions


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor`
            # and `int`.
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),  # noqa E501
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor`
            # and `int`.
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),  # noqa E501
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor`
            # and `int`.
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),  # noqa E501
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor`
            # and `int`.
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),  # noqa E501
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is
    # small, the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to
    # a sign), forall i; we pick the best-conditioned one (with the largest
    # denominator)
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


# # ======== OLD ========
def normalize(
    seq,
    frame: int,
    jid_left=6,
    jid_right=12,
    return_transform=False,
    allow_zero_z=False,
    fix_shape=False,
):
    """
    :param seq: {n_frames x J x 3}
    :param frame:
    """
    if fix_shape:
        # seq = unknown_pose_shape_to_known_shape(seq)
        raise NotImplementedError("Not yet..")
    assert len(seq) > frame, f"frame:{frame} but shape: {seq.shape}"
    left3d = seq[frame, jid_left]
    right3d = seq[frame, jid_right]

    if not allow_zero_z:
        if np.isclose(left3d[2], 0.0):
            raise ValueError(f"Left seems to be zero! {left3d}")
        if np.isclose(right3d[2], 0.0):
            raise ValueError(f"Right seems to be zero! {right3d}")

    mu, R = get_normalization(left3d, right3d)
    if return_transform:
        return apply_normalization_to_seq(seq, mu, R), (mu, R)
    else:
        return apply_normalization_to_seq(seq, mu, R)


# def apply_normalization_to_seq(seq, mu, R, fix_shape=False):
#     """
#     :param seq: {n_frames x 18 x 3}
#     :param mu: {3}
#     :param R: {3x3}
#     """
#     mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
#     if fix_shape:
#         raise NotImplementedError("nope")
#         # seq = unknown_pose_shape_to_known_shape(seq)
#     seq = seq - mu
#     return rot.apply_rotation_to_seq(seq, R, fix_shape=fix_shape)


# def unknown_pose_shape_to_known_shape(seq):  # noqa: C901
#     """
#     Sometimes we don't know what the pose is shaped like:
#     We could get either a single pose: (18*3) OR (18x3) OR (17*3) OR (17x3)
#     OR it could be a sequence: (nx18*3) OR (nx18x3) OR (nx17*3) OR (nx17x3)
#     This function takes any of those pose(s) and transforms
#     into (nx18x3) OR (nx17x3)
#     """
#     if isinstance(seq, list):
#         seq = np.array(seq, dtype=np.float32)
#     if len(seq.shape) == 1:
#         # has to be dimension 18*3 or 17*3
#         if seq.shape[0] == 18 * 3:
#             seq = seq.reshape(1, 18, 3)
#         else:
#             seq = seq.reshape(1, 17, 3)
#     elif len(seq.shape) == 2:
#         if seq.shape[0] == 18:
#             if seq.shape[1] != 3:
#                 raise ValueError("(1) Incorrect shape:" + str(seq.shape))
#             seq = seq.reshape(1, 18, 3)
#         elif seq.shape[0] == 17:
#             if seq.shape[1] != 3:
#                 raise ValueError("(1) Incorrect shape:" + str(seq.shape))
#             seq = seq.reshape(1, 17, 3)
#         else:
#             if seq.shape[1] != 18 * 3 and seq.shape[1] != 17 * 3:
#                 raise ValueError("(2) Incorrect shape:" + str(seq.shape))
#             n = len(seq)

#             if seq.shape[1] == 18 * 3:
#                 seq = seq.reshape(n, 18, 3)
#             else:
#                 seq = seq.reshape(n, 17, 3)
#     else:
#         if len(seq.shape) != 3:
#             raise ValueError("(3) Incorrect shape:" + str(seq.shape))
#         if (seq.shape[1] != 18 and seq.shape[1] != 17) or seq.shape[2] != 3:
#             raise ValueError("(4) Incorrect shape:" + str(seq.shape))
#     return seq


def to_canconical_form(seq):
    """
    :param seq: {n_frames x 57}
    :return: {n_frames x (57 + 2*3)}
    """
    was_numpy = False
    if isinstance(seq, np.ndarray):
        seq = torch.from_numpy(seq)
        was_numpy = True
    seq = rearrange(seq, "t (j d) -> t j d", j=19, d=3)
    R, mu = tn_get_normalization_for_pose(seq, left_jid=6, right_jid=12)
    seq_canonical = tn_framewise_transform(seq, mu, R)
    hips_world = seq[:, [6, 12]]
    seq_canonical = torch.cat([seq_canonical, hips_world], dim=1)
    if was_numpy:
        seq_canonical = seq_canonical.numpy()
    seq_canonical = rearrange(seq_canonical, "t j d -> t (j d)")
    return seq_canonical


def tn_get_normalization_for_pose(Pose, left_jid=13, right_jid=14):
    """
    0 1 2 3 4
    5 6 7 8 9 10
    11 12 [13 14] 15 16
    :param Pose: {n_batch x 17 x 3}
    """
    n_batch = len(Pose)
    if len(Pose.shape) == 2:
        # J = Pose.size(1) // 3
        Pose = Pose.reshape(n_batch, 17, 3)
    assert len(Pose.shape) == 3, f"wrong shape: {Pose.shape}"
    left3d = Pose[:, left_jid]
    right3d = Pose[:, right_jid]
    return tn_get_normalization(left3d=left3d, right3d=right3d)