import numpy as np
from typing import List, Tuple
from social_diffusion import WeirdShapeError
from einops import rearrange
import torch

from social_diffusion.transforms.transforms import (
    matrix_to_axis_angle,
    tn_batch_normalize,
    tn_batch_rvec_to_rotpoint,
    tn_batch_rotpoint_to_rvec,
    tn_framewise_undo_transform,
)
from social_diffusion.transforms.transforms import axis_angle_to_matrix


class AbstractSkeleton:
    def __init__(
        self,
        is_local: bool,
        loss_weight: np.ndarray = None,
        leg_joints: List[int] = None,
    ):
        self.is_local = is_local
        if loss_weight is None:
            self.loss_weight = np.ones((self.jd()), dtype=np.float32)
        else:
            if len(loss_weight.shape) != 1 and len(loss_weight) != self.jd():
                raise WeirdShapeError(f"Weird loss weights: {loss_weight.shape}")
            self.loss_weight = loss_weight

        if leg_joints is not None:
            leg_joints = np.array(leg_joints, dtype=np.int64)
            if len(leg_joints.shape) != 1:
                raise WeirdShapeError(
                    f"Weird leg-joints shape: {leg_joints.shape}, req. 1d"
                )
        self.leg_joints = leg_joints

    def __len__(self):
        raise NotImplementedError("Nope")

    def get_leg_joints(self):
        if self.leg_joints is None:
            raise NotImplementedError("Nope...")
        else:
            return self.leg_joints

    def get_loss_weight(self):
        return self.loss_weight

    def jd(self):
        raise NotImplementedError("Nope")

    def from_local(self, device=None, use_rotpoints=True):
        raise NotImplementedError("Nope")

    def to_local(self, seq, device=None, is_flat=None, use_rotpoints=True):
        raise NotImplementedError("Nope")

    def jd_without_normalization(self):
        return self.jd()

    def from_local_group(self, Seq, device=None, use_rotpoints=True):
        """
        :param Seq: {n_frames x n_person x 6+jd}
        """
        n_person = Seq.shape[1]
        Seq_rec = []
        for pid in range(n_person):
            Seq_rec.append(
                self.from_local(Seq[:, pid], device=device, use_rotpoints=use_rotpoints)
            )
        return rearrange(Seq_rec, "p t jd -> t p jd")

    def to_local_group(self, Seq, device=None, is_flat=None, use_rotpoints=True):
        """
        :param Seq: {n_frames x n_person x 6+jd}
        """
        n_person = Seq.shape[1]
        Seq_loc = []
        for pid in range(n_person):
            Seq_loc.append(
                self.to_local(
                    Seq[:, pid],
                    device=device,
                    use_rotpoints=use_rotpoints,
                    is_flat=is_flat,
                )
            )
        return rearrange(Seq_loc, "p t jd -> t p jd")

    def fix_shape(self, seq, unroll_jd=False):
        """
        :param: seq: {n_frames x n_joints * 3}
        """
        n_joints = len(self)
        if len(seq.shape) == 1:
            if seq.shape[0] != self.jd():
                raise WeirdShapeError(f"(a) Weird shape: {seq.shape}")
            seq = rearrange(seq, "jd -> 1 jd")
        elif len(seq.shape) == 2:
            if seq.shape[1] == 3:
                if seq.shape[0] != n_joints:
                    raise WeirdShapeError(f"(a/1) Weird shape: {seq.shape}")
                seq = rearrange(seq, "j d -> 1 (j d)")
            elif seq.shape[1] == self.jd():
                seq = seq
            else:
                raise WeirdShapeError(f"(b) Weird shape: {seq.shape}, jd:{self.jd()}")
        elif len(seq.shape) == 3:
            if seq.shape[2] == 3:
                if seq.shape[1] != n_joints:
                    raise WeirdShapeError(
                        f"seq:{seq.shape} does not fit n_joints:{n_joints}"
                    )
                seq = rearrange(seq, "t j d -> t (j d)")
            else:
                raise WeirdShapeError(f"(c) Weird shape: {seq.shape}")
        if unroll_jd:
            return rearrange(seq, "t (j d) -> t j d", j=n_joints)
        else:
            return seq

    def split_local(self, seq):
        raise NotImplementedError("Nope")


class Skeleton(AbstractSkeleton):
    def __init__(
        self,
        normalizing_left_jid: int,
        normalizing_right_jid: int,
        n_joints: int,
        skeleton: List[Tuple[int, int]],
        left_jids: List[int],
        right_jids: List[int],
        *,
        standardization_mask: np.ndarray = None,
        loss_weight: np.ndarray = None,
        leg_joints: List[int] = None,
    ):
        self.normalizing_left_jid = normalizing_left_jid
        self.normalizing_right_jid = normalizing_right_jid
        self.n_joints = n_joints
        self.skeleton = skeleton
        self.left_jids = left_jids
        self.right_jids = right_jids
        self.standardization_mask = (
            standardization_mask
            if standardization_mask is not None
            else np.ones((1, self.jd()), dtype=np.float32)
        )
        super().__init__(is_local=False, loss_weight=loss_weight, leg_joints=leg_joints)

    def __len__(self):
        return self.n_joints

    def jd(self):
        return self.n_joints * 3

    def flip_lr(self, seq):
        """
        :param: seq: {n_frames x n_joints * 3}
        """
        seq = self.fix_shape(seq, unroll_jd=True)
        LR = np.array(list(self.left_jids) + list(self.right_jids), dtype=np.int64)
        RL = np.array(list(self.right_jids) + list(self.left_jids), dtype=np.int64)
        if len(LR) != 2 * len(self.left_jids):
            raise WeirdShapeError(
                f"Weird shape: {len(LR)} vs {len(self.left_jids)} (left)"
            )
        if len(RL) != 2 * len(self.right_jids):
            raise WeirdShapeError(
                f"Weird shape: {len(LR)} vs {len(self.right_jids)} (right)"
            )
        seq[:, LR] = seq[:, RL]
        I = np.diag([1.0, -1.0, 1.0])  # noqa: E741
        seq_flat = rearrange(seq, "t j d -> (t j) d")
        seq_flat = seq_flat @ I
        seq = rearrange(seq_flat, "(t j) d -> t j d", j=self.n_joints)
        return np.ascontiguousarray(rearrange(seq, "t j d -> t (j d)"))

    def to_local(self, seq, device=None, is_flat=None, use_rotpoints=True):
        """
        Poses are centered at the hip
        :param seq: {n_frames x jd}
        :param use_rotpoints: if True we use rotation points instead of rvec
            rotation points do not suffer from discontinuities
        :returns:
            {n_frames x 3 + 3 + jd}
            Mu + Rvec + jd
        """
        if is_flat is None:
            is_flat = len(seq.shape) == 2
        is_numpy = isinstance(seq, np.ndarray)
        seq = self.fix_shape(seq, unroll_jd=True)
        seq, R, Mu = tn_batch_normalize(
            Pose=seq, skel=self, return_transform=True, device=device
        )
        Rvec = matrix_to_axis_angle(R)
        if use_rotpoints:
            Rvec = tn_batch_rvec_to_rotpoint(Rvec, device=device)
        Rvec = rearrange(Rvec, "t d -> t 1 d")
        Mu = rearrange(Mu, "t d -> t 1 d")
        seq = torch.cat([Mu, Rvec, seq], dim=1)
        if is_numpy:
            seq = seq.cpu().numpy()
        if is_flat:
            seq = rearrange(seq, "t j d -> t (j d)")
        return seq

    def from_local(self, seq, device=None, use_rotpoints=True):
        """
        Poses are centered at the hip
        :param seq: {n_frames x jd}
        :returns:
            {n_frames x 3 + 3 + jd}
            Mu + Rvec + jd
        """
        is_flat = len(seq.shape) == 2
        is_numpy = isinstance(seq, np.ndarray)
        if is_numpy:
            seq = torch.from_numpy(seq)
        if device is None:
            device = torch.device("cpu")
        seq = seq.to(device)
        if is_flat:
            seq = rearrange(seq, "t (j d) -> t j d", d=3)
        if (
            len(seq.shape) != 3
            or seq.shape[1] != self.n_joints + 2
            or seq.shape[2] != 3
        ):
            raise WeirdShapeError(f"Weird shape: {seq.shape}, note that j=j+2")
        seq, Mu, Rvec = self.split_local(seq)
        if use_rotpoints:
            Rvec = tn_batch_rotpoint_to_rvec(Rvec, device)
        R = axis_angle_to_matrix(Rvec)
        seq = tn_framewise_undo_transform(poses=seq, mu=Mu, R=R)
        if is_numpy:
            seq = seq.cpu().numpy()
        if is_flat:
            seq = rearrange(seq, "b j d -> b (j d)")
        return seq

    def split_local(self, seq):
        """
        :param seq: {n_frames x jd}
        :returns:
            seq, rvec, mu
        """
        is_flat = False
        if len(seq.shape) == 2:
            seq = rearrange(seq, "t (j d) -> t j d", d=3)
            is_flat = True
        if (
            len(seq.shape) != 3
            or seq.shape[1] != self.n_joints + 2
            or seq.shape[2] != 3
        ):
            raise WeirdShapeError(
                f"Weird shape for splitting: {seq.shape}, J={self.n_joints}"
            )
        mu = seq[:, 0]
        rvec = seq[:, 1]
        poses = seq[:, 2:]
        if is_flat:
            poses = rearrange(poses, "t j d -> t (j d)")
        return poses, mu, rvec


class LocalSkeleton(AbstractSkeleton):
    """
    LocalSkeleton has the pose in canonical space and
    mu and rvec as first and second entry!
    [mu, rvec, pose]
    """

    def __init__(self, parent_skel: Skeleton, loss_weight: np.ndarray = None):
        if loss_weight is None:
            jd = parent_skel.jd() + 6
            loss_weight = np.ones((jd), dtype=np.float32)
            # loss_weight[:6] = 0.0
            # loss_weight[:3] = 1.0
            # loss_weight[3:] = 0

        self.normalizing_left_jid = parent_skel.normalizing_left_jid + 2
        self.normalizing_right_jid = parent_skel.normalizing_right_jid + 2
        self.n_joints = parent_skel.n_joints + 2

        leg_joints = None
        if parent_skel.leg_joints is not None:
            leg_joints = parent_skel.get_leg_joints() + 2

        self.parent_skel = parent_skel
        self.left_jids = (np.array(parent_skel.left_jids, dtype=np.int64) + 2).tolist()
        self.right_jids = (
            np.array(parent_skel.right_jids, dtype=np.int64) + 2
        ).tolist()
        self.skeleton = (np.array(parent_skel.skeleton, dtype=np.int64) + 2).tolist()
        self.standardization_mask = np.ones((1, self.jd()), dtype=np.float32)
        self.standardization_mask[0, :3] = 0
        super().__init__(is_local=True, loss_weight=loss_weight, leg_joints=leg_joints)

    def __len__(self):
        return self.n_joints

    def fix_shape(self, seq, unroll_jd=False):
        try:
            return super().fix_shape(seq, unroll_jd=unroll_jd)
        except WeirdShapeError:
            return self.parent_skel.fix_shape(seq, unroll_jd=unroll_jd)

    def jd_without_normalization(self):
        return self.jd() - 6

    def jd(self):
        return self.n_joints * 3

    def split_local(self, seq):
        return self.parent_skel.split_local(seq)

    def to_local(self, seq, *, use_rotpoints=True, is_flat=None, device=None):
        return self.parent_skel.to_local(
            seq, use_rotpoints=use_rotpoints, is_flat=is_flat, device=device
        )

    def from_local(self, seq, device=None, use_rotpoints=True):
        return self.parent_skel.from_local(
            seq, device=device, use_rotpoints=use_rotpoints
        )