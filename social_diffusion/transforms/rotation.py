import numpy as np
import torch


def tn_rot3d_c(c):
    """
    Rotation only around the z axis!
    :param c: {n_batch x 1}
    """
    if len(c.shape) == 1:
        c = c.unsqueeze(1)

    z = torch.zeros_like(c)
    o = torch.ones_like(c)

    cos_c = torch.cos(c)  # n_batch x 1
    sin_c = torch.sin(c)

    Rz_l1 = torch.cat([cos_c, -sin_c, z], dim=1).unsqueeze(1)
    Rz_l2 = torch.cat([sin_c, cos_c, z], dim=1).unsqueeze(1)
    Rz_l3 = torch.cat([z, z, o], dim=1).unsqueeze(1)

    Rz = torch.cat([Rz_l1, Rz_l2, Rz_l3], dim=1)

    return Rz


def rot3d(a, b, c):
    """"""
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]],
        np.float32,
    )
    Ry = np.array(
        [[np.cos(b), 0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0, np.cos(b)]],
        np.float32,
    )
    Rz = np.array(
        [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    return np.ascontiguousarray(Rx @ Ry @ Rz)


def apply_rotation_to_seq(seq, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param R: {3x3}
    """
    R = np.expand_dims(R, axis=0)
    return np.ascontiguousarray(seq @ R)