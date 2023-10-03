import logging
import math as m
from tqdm import tqdm
from social_diffusion.transforms.rotation import rot3d
import social_diffusion.transforms.transforms as T
from social_diffusion import get_cache_dir, WeirdShapeError
from social_diffusion.sequence import (
    MultiPersonSequence,
    sequence_stats_to_unique_str,
)  # noqa E501
from social_diffusion.datasets import calculate_standardization
from social_diffusion.skeleton import LocalSkeleton
from numpy.random import uniform
from torch.utils.data import Dataset
from typing import List
import numpy as np
from einops import rearrange, reduce
from os.path import join, isfile
import hashlib
import torch


logger = logging.getLogger(__name__)


class MultiPersonDataset(Dataset):
    def __init__(
        self,
        sequences: List[MultiPersonSequence],
        n_frames: int,
        *,
        ss: int = 10,
        create_batch_entry_fn=None,
        undo_batch_fn=None,
        augmentation_fn=None,
        do_caching: bool = True,
        do_standardize: bool = True,
        p_mu=None,
        p_std=None,
        skel=None,
        calculate_standardization_fn=None,
    ):
        """
        :param create_batch_entry_fn:
            fn(P: {n_frames x n_person x jd}, skel: Skeleton) -> {batch entry}
            undo_data
        :param undo_batch_fn: fn(P: {n_frames x n_person x jd}
            kel: Skeleton, undo_data)
        :param augmentation_fn: fn(Seq: n_frames x n_person x jd}
            skel: Skeleton)
        """
        super().__init__()

        self.create_batch_entry_fn = create_batch_entry_fn
        self.undo_batch_fn = undo_batch_fn
        if augmentation_fn is None:
            self.augmentation_fn = lambda Seq, skel: Seq
        else:
            self.augmentation_fn = augmentation_fn

        if p_mu is not None and not do_standardize:
            raise ValueError(
                "You pass standardization parameter but you have standardization deactivated.. bug?"  # noqa E501
            )

        self.do_standardize = do_standardize
        if skel is None:
            self.skel = sequences[0].skel
        else:
            self.skel = skel

        self.__data = []
        n_dummy_person = sequences[0].n_max_person()
        cache_fname = None
        if do_caching:
            # --- create cache_fname ---
            cache_fname = (
                MultiPersonSequence.global_uid(sequences)
                + f"_nframes:{n_frames}_ss:{ss}"
            )
            if create_batch_entry_fn is not None:
                # we pass a dummy sequence to the preprocess_fn
                dummy = np.empty(
                    (
                        n_frames,
                        n_dummy_person,
                        self.skel.jd_without_normalization(),
                    ),  # noqa E501
                    dtype=np.float32,
                )
                incr_t = rearrange(
                    np.linspace(start=-1.0, stop=1.0 + n_frames, num=n_frames),
                    "t -> t 1 1",
                )
                incr_p = rearrange(
                    np.linspace(
                        start=-2.0, stop=n_dummy_person, num=n_dummy_person
                    ),  # noqa E501
                    "p -> 1 p 1",
                )
                for jid in range(dummy.shape[2]):
                    dummy[:, :, jid] = jid
                dummy = incr_p + dummy * incr_t
                dummy += 0.000001
                dummy_norm, _ = create_batch_entry_fn(dummy, self.skel)
                prep_str = sequence_stats_to_unique_str(
                    rearrange(dummy_norm, "t p jd -> (t p) jd")
                )
                cache_fname += prep_str

            cache_fname = join(
                get_cache_dir(),
                "mpd_"
                + hashlib.sha1(cache_fname.encode("utf-8")).hexdigest()
                + ".npy",  # noqa E501
            )

            if isfile(cache_fname):
                self.__data = np.load(cache_fname, mmap_mode="r")

        if len(self.__data) == 0:
            for seq in tqdm(
                sequences, total=len(sequences), position=0, leave=True
            ):  # noqa E501
                n_person = seq.n_max_person()
                if seq.is_masked():
                    raise ValueError("We do not allow masked sequences!")
                if n_person != n_dummy_person:
                    raise ValueError(
                        "We expect all sequences to have the same amount of persons!"  # noqa E501
                    )
                for t in range(0, len(seq) - n_frames + 1, ss):
                    P, _ = self.create_batch_entry(
                        seq.Seq[t : t + n_frames].copy()  # noqa E501
                    )  # noqa E501
                    self.__data.append(P)

            self.__data = np.array(self.__data, dtype=np.float32)
            if do_caching:
                np.save(cache_fname, self.__data)
        self.p_mu = None
        self.p_std = None
        if do_standardize:
            if p_mu is None:
                assert p_std is None

                # calculate standardization
                if calculate_standardization_fn is None:
                    logger.info("utilize default standardization function")
                    calculate_standardization_fn = calculate_standardization
                else:
                    logger.info("utilize custom standardization function")

                cache_fname_std = None
                if do_caching:
                    assert cache_fname is not None
                    cache_fname_std = cache_fname.replace(
                        ".npy", "_standard.npz"
                    )  # noqa E501
                    if isfile(cache_fname_std):
                        standard = np.load(cache_fname_std)
                        self.p_mu = standard["p_mu"]
                        self.p_std = standard["p_std"]
                if self.p_mu is None:
                    assert self.p_std is None
                    # calculate standardization
                    p_mu, p_std = calculate_standardization_fn(
                        self.__data, self.skel
                    )  # noqa E501
                    self.p_mu = rearrange(p_mu, "jd -> 1 jd").astype(
                        np.float32
                    )  # noqa E501
                    self.p_std = rearrange(p_std, "jd -> 1 jd").astype(
                        np.float32
                    )  # noqa E501
                    if do_caching:
                        np.savez(
                            cache_fname_std, p_mu=self.p_mu, p_std=self.p_std
                        )  # noqa E501
            else:
                assert p_std is not None
                self.p_mu = p_mu
                self.p_std = p_std

            # mask out standardization
            standardization_mask = skel.standardization_mask  # 1 x jd
            inverse_standardization_mask = (
                (standardization_mask < 0.5) * 1
            ).astype(  # noqa E501
                "float32"
            )
            fixed_p_std = np.ones_like(self.p_std)
            self.p_mu = self.p_mu * standardization_mask
            self.p_std = (
                self.p_std * standardization_mask
                + fixed_p_std * inverse_standardization_mask
            )

        # -- ctor over --

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index: int):
        return {
            "X_0": self.do_standardization(
                self.augmentation_fn(self.__data[index], self.skel)
            )
        }

    def get_raw(self, index: int):
        return self.__data[index]

    def undo_batch_entry(self, P, undo_data=None):
        if self.undo_batch_fn is None:
            return P
        else:
            return self.undo_batch_fn(P, self.skel, undo_data)

    def create_batch_entry(self, P):
        if self.create_batch_entry_fn is None:
            return P, None
        else:
            return self.create_batch_entry_fn(P, self.skel)

    def do_standardization(self, Seq):
        """
        :param Seq: {n_frames x 3 x jd}
        """
        if self.do_standardize:
            if not isinstance(Seq, np.ndarray):
                Seq = Seq.detach().cpu().numpy()
            assert Seq.shape[2] == self.skel.n_joints * 3
            p_mu = rearrange(self.p_mu, "b jd -> 1 b jd")
            p_std = rearrange(self.p_std, "b jd -> 1 b jd")
            return (Seq - p_mu) / p_std
        else:
            return Seq

    def undo_standardization(self, Seq):
        """
        :param Seq: {n_frames x 3 x jd}
        """
        if self.do_standardize:
            if not isinstance(Seq, np.ndarray):
                Seq = Seq.detach().cpu().numpy()
            assert Seq.shape[2] == self.skel.n_joints * 3
            p_mu = rearrange(self.p_mu, "b jd -> 1 b jd")
            p_std = rearrange(self.p_std, "b jd -> 1 b jd")
            return p_mu + (Seq * p_std)
        else:
            return Seq

    def batch_undo_standardization(self, Seq):
        """
        Keep gradients!
        :param Seq: {n_batch x n_frames x 3 x jd}
        """
        if self.do_standardize:
            p_mu = torch.from_numpy(
                rearrange(self.p_mu, "b jd -> 1 1 b jd")
            ).to(  # noqa E501
                Seq.device
            )
            p_std = torch.from_numpy(
                rearrange(self.p_std, "b jd -> 1 1 b jd")
            ).to(  # noqa E501
                Seq.device
            )
            return p_mu + (Seq * p_std)
        else:
            return Seq


def get_NormalizedMultiPersonDataset(
    sequences: List[MultiPersonSequence],
    n_frames: int,
    normalize_at_frame: int,
    *,
    ss: int = 10,
    do_caching: bool = True,
    do_standardize=True,
    # ignore_warnings=False,
) -> MultiPersonDataset:
    # JULIAN
    def create_batch_entry_fn(Seq, skel):
        if len(Seq.shape) != 3:
            raise WeirdShapeError(
                f"Weird shape for normalization: {Seq.shape}"
            )  # noqa E501

        n_person = Seq.shape[1]

        Seq_norm = []
        Mus = []
        Rs = []
        for pid in range(n_person):
            seq = Seq[:, pid]
            seq = skel.fix_shape(seq, unroll_jd=True)
            seq_norm, (mu, R) = T.normalize(
                seq,
                jid_left=skel.normalizing_left_jid,
                jid_right=skel.normalizing_right_jid,
                frame=normalize_at_frame,
                return_transform=True,
            )
            seq_norm = rearrange(seq_norm, "t j d -> t (j d)")
            Seq_norm.append(seq_norm)
            Mus.append(mu)
            Rs.append(R)

        Seq_norm = np.array(Seq_norm, dtype=np.float32)
        Seq_norm = rearrange(Seq_norm, "p t jd -> t p jd")

        return Seq_norm, {"Mus": Mus, "Rs": Rs}

    def undo_batch_fn(Seq, skel, undo_data):
        """
        :param Seq: {n_frames x n_person x jd}
        :param undo_data: {
            "Mus"
            "Rs"
        }
        """
        if len(Seq.shape) != 3:
            raise WeirdShapeError(
                f"Weird shape for normalization: {Seq.shape}"
            )  # noqa E501

        Mus = undo_data["Mus"]
        Rs = undo_data["Rs"]
        n_person = Seq.shape[1]

        Seq_recover = []
        for pid in range(n_person):
            seq = Seq[:, pid]
            Seq_recover.append(
                T.undo_normalization_to_seq(seq=seq, mu=Mus[pid], R=Rs[pid])
            )
        Seq_recover = np.array(Seq_recover, dtype=np.float32)

        # print("Seq_recover", Seq_recover.shape)
        return Seq_recover

    return MultiPersonDataset(
        sequences=sequences,
        n_frames=n_frames,
        ss=ss,
        do_caching=do_caching,
        do_standardize=do_standardize,
        # ignore_warnings=ignore_warnings,
        skel=sequences[0].skel,
        create_batch_entry_fn=create_batch_entry_fn,
        undo_batch_fn=undo_batch_fn,
    )


def get_LocalMultiPersonDataset(
    sequences: List[MultiPersonSequence],
    n_frames: int,
    normalize_at_frame: int,
    *,
    ss: int = 10,
    do_caching: bool = True,
    p_mu=None,
    p_std=None,
    do_standardize=True,
    # ignore_warnings=False,
    use_rotpoints=True,
    normalize=False,
    augmentation_random_rotate=True,
    augmentation_random_translate=False,
) -> MultiPersonDataset:
    def create_batch_entry_fn(Seq, skel):
        if len(Seq.shape) != 3:
            raise WeirdShapeError(
                f"Weird shape for normalization: {Seq.shape}"
            )  # noqa E501

        n_person = Seq.shape[1]

        Seq_norm = []
        for pid in range(n_person):
            seq = Seq[:, pid]
            seq = skel.to_local(seq, use_rotpoints=use_rotpoints, is_flat=True)
            Seq_norm.append(seq)
        Seq_norm = np.array(Seq_norm, dtype=np.float32)
        Seq_norm = rearrange(Seq_norm, "p t jd -> t p jd")
        # --- v ---
        # print("== WARNING == WE DESTROY POSE + ROTATION HERE!")
        # t, p, jd = Seq_norm.shape
        # Seq_norm[:, :, 3:] = np.random.uniform(
        #     low=-0.01, high=0.01, size=(t, p, jd - 3)
        # )
        # print("== WARNING == WE DESTROY POSE + ROTATION HERE!")
        # --- ^ ---

        if normalize:
            # Last_loc_for_all = reduce(
            #     Seq_norm[normalize_at_frame - 1, :, 0:3], "p d -> d", "mean"
            # )
            # Last_loc_for_all[2] = 0
            # t = rearrange(Last_loc_for_all, "d -> 1 1 d")
            t = rearrange(Seq_norm[normalize_at_frame, :, :3], "p d -> 1 p d")
            # Seq_norm[:, :, :3] = Seq_norm[:, :, :3] - t
            # return Seq_norm, {"translate": t}
            Seq_norm[:, :, :3] = Seq_norm[:, :, :3] - t
            return Seq_norm, {"translate": t}
        else:
            return Seq_norm, None

    def undo_batch_fn(Seq, skel, undo_data):
        """
        :param Seq: {n_frames x n_person x jd}
        :param undo_data: {
            "translate", 1x3 for each n_person
        }
        """
        if len(Seq.shape) != 3:
            raise WeirdShapeError(
                f"Weird shape for normalization: {Seq.shape}"
            )  # noqa E501

        if undo_data is not None:
            t = undo_data["translate"]
            Seq[:, :, :3] += t

        return skel.from_local_group(Seq, use_rotpoints=use_rotpoints)

    def augmentation_fn(Seq, skel):
        if (
            not augmentation_random_rotate
            and not augmentation_random_translate  # noqa E501
        ):  # noqa E501
            return Seq
        else:
            if augmentation_random_rotate:
                if len(Seq.shape) != 3:
                    raise WeirdShapeError(f"Weird shape: {Seq.shape}")
                angle = uniform(low=0.0, high=m.pi * 2)
            else:
                angle = 0

            R = rot3d(0.0, 0.0, angle)

            if augmentation_random_translate:
                t = uniform(low=-3.0, high=3.0, size=(1, 1, 1, 3))
                t[:, :, :, 2] = 0
            else:
                t = np.zeros((1, 1, 1, 3), dtype=np.float32)

            n_person = Seq.shape[1]
            Seq_out = []
            for pid in range(n_person):
                seq = skel.fix_shape(
                    Seq[:, pid].copy(), unroll_jd=True
                )  # n_frames x j x 3
                seq[:, 0] = seq[:, 0] @ R + t
                seq[:, 1] = seq[:, 1] @ R  # WE ASSUME THIS IS rotpoint!
                Seq_out.append(seq)
            return rearrange(Seq_out, "p t j d -> t p (j d)")

    calculate_standardization_fn = None
    if augmentation_random_rotate or augmentation_random_translate:
        # if we randomly rotate (augment) we HAVE to adjust the standardization
        # function!

        def calculate_standardization_fn(all_poses, skel):
            """
            :param all_poses: {b t p jd}
            """
            eps = 0.000000001
            assert len(all_poses.shape) == 4, f"weird shape: {all_poses.shape}"
            assert skel.is_local
            all_poses = rearrange(
                all_poses, "b t p (j d) -> b t p j d", j=skel.n_joints
            )

            loc = rearrange(all_poses[:, :, :, 0], "b t p d -> (b t p) d")
            rot = rearrange(all_poses[:, :, :, 1], "b t p d -> (b t p) d")
            pose = rearrange(
                all_poses[:, :, :, 2:], "b t p j d -> (b t p) (j d)"
            )  # noqa E501

            Locs = []
            Rots = []
            if augmentation_random_rotate:
                for angle in np.linspace(start=0, stop=2 * m.pi, num=32):
                    R = rot3d(0.0, 0.0, angle)
                    Locs.append(loc.copy() @ R)
                    Rots.append(rot.copy() @ R)
            else:
                Rots.append(rot)

            if augmentation_random_translate:
                for x in np.linspace(start=-3, stop=3, num=10):
                    for y in np.linspace(start=-3, stop=3, num=10):
                        t = np.array([x, y, 0], dtype=np.float32)
                        Locs.append(loc.copy() + t)
            else:
                Locs.append(loc)

            Locs = np.concatenate(Locs, axis=0)
            Rots = np.concatenate(Rots, axis=0)

            p_mu_pose = reduce(pose, "t jd -> jd", "mean")
            p_std_pose = np.std(pose, axis=0) + eps

            p_mu_loc = reduce(Locs, "t jd -> jd", "mean")
            p_std_loc = np.std(Locs, axis=0) + eps

            p_mu_rot = reduce(Rots, "t jd -> jd", "mean")
            p_std_rot = np.std(Rots, axis=0) + eps

            p_mu = np.concatenate([p_mu_loc, p_mu_rot, p_mu_pose], axis=0)
            p_std = np.concatenate([p_std_loc, p_std_rot, p_std_pose], axis=0)

            return p_mu, p_std

    parent_skel = sequences[0].skel
    skel = LocalSkeleton(parent_skel)
    return MultiPersonDataset(
        sequences=sequences,
        n_frames=n_frames,
        ss=ss,
        do_caching=do_caching,
        p_mu=p_mu,
        p_std=p_std,
        do_standardize=do_standardize,
        # ignore_warnings=ignore_warnings,
        skel=skel,
        create_batch_entry_fn=create_batch_entry_fn,
        undo_batch_fn=undo_batch_fn,
        augmentation_fn=augmentation_fn,
        calculate_standardization_fn=calculate_standardization_fn,
    )


def get_LocalOnlyMultiPersonDataset(
    sequences: List[MultiPersonSequence],
    n_frames: int,
    ss: int = 10,
    do_caching: bool = True,
    p_mu=None,
    p_std=None,
    do_standardize=True,
    # ignore_warnings=False,
):
    """
    This is a **debugging** module! Only use local rep!
    """

    def create_batch_entry_fn(Seq, skel):
        if len(Seq.shape) != 3:
            raise WeirdShapeError(
                f"Weird shape for normalization: {Seq.shape}"
            )  # noqa E501

        n_person = Seq.shape[1]

        Seq_norm = []
        for pid in range(n_person):
            seq = Seq[:, pid]
            seq = skel.to_local(seq, use_rotpoints=True, is_flat=True)
            Seq_norm.append(seq)
        Seq_norm = np.array(Seq_norm, dtype=np.float32)
        Seq_norm = rearrange(Seq_norm, "p t (j d) -> t p j d", d=3)
        Seq_norm = rearrange(Seq_norm[:, :, 2:], "t p j d -> t p (j d)")
        return Seq_norm, None

    skel = sequences[0].skel
    return MultiPersonDataset(
        sequences=sequences,
        n_frames=n_frames,
        ss=ss,
        do_caching=do_caching,
        p_mu=p_mu,
        p_std=p_std,
        do_standardize=do_standardize,
        # ignore_warnings=ignore_warnings,
        skel=skel,
        create_batch_entry_fn=create_batch_entry_fn,
    )