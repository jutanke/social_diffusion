import hashlib
from multiprocessing.pool import ThreadPool
from einops import rearrange, reduce
from typing import List
import numpy as np
import numba as nb
from social_diffusion import WeirdShapeError
from social_diffusion.skeleton import Skeleton


class Sequence:
    def __init__(self, n_frames):
        self.n_frames = n_frames

    def get_uid(self) -> str:
        raise NotImplementedError()

    def __len__(self):
        return self.n_frames


def get_uid(seq):
    return seq.get_uid()


class MultiPersonSequence(Sequence):
    """
    Represents an entire continous sequence of potentially multiple persons
    A mask defines if a person is visible (1) or not (0)
    """

    @staticmethod
    def global_uid(sequences: List):
        with ThreadPool(processes=len(sequences)) as p:
            return (
                "Gmps_"
                + hashlib.sha1(  # noqa B324
                    "___".join(p.map(get_uid, sequences)).encode("utf-8")
                ).hexdigest()
            )

    def __init__(
        self, Seq: np.ndarray, skel: Skeleton, Mask: np.ndarray = None
    ):  # noqa E501
        """
        :param Seq: {n_frames x n_person x jd}
        :param Mask: {n_frames x n_person} in {0, 1}
        """
        self.__uid = None
        self.n_frames = Seq.shape[0]
        self.n_person = Seq.shape[1]
        if Mask is None:
            Mask = np.ones((self.n_frames, self.n_person), dtype=np.float32)
        else:
            if (
                len(Mask.shape) != 2
                or Mask.shape[0] != self.n_frames
                or Mask.shape[1] != self.n_person
            ):
                raise WeirdShapeError(f"weird mask shape: {Mask.shape}")

        if len(Seq.shape) == 4:
            if Mask.shape[3] != 3:
                raise WeirdShapeError(f"(1) weird sequence shape: {Seq.shape}")
            if Seq.shape[2] != skel.n_joints:
                raise WeirdShapeError(
                    f"(1) Seq shape ({Seq.shape}) and Skeleton (j={skel.n_joints}) do not fit"  # noqa E501
                )
            Seq = rearrange("t p j d -> t p (j d)")
        if len(Seq.shape) != 3:
            raise WeirdShapeError(f"(2) weird sequence shape: {Seq.shape}")
        if Seq.shape[2] != 3 * skel.n_joints:
            f"(2) Seq shape ({Seq.shape}) and Skeleton (j={skel.n_joints}) do not fit"  # noqa E501

        super().__init__(n_frames=self.n_frames)

        self.skel = skel

        if Seq.dtype != np.float32:
            Seq = Seq.astype(np.float32)
        if Mask.dtype != np.float32:
            Mask = Mask.astype(np.float32)
        self.Seq = Seq * rearrange(Mask, "t p -> t p 1")
        self.Mask = Mask

    def flip_lr(self):
        """
        flips left-right jids
        """
        Seq_new = self.Seq.copy()
        Mask_new = self.Mask.copy()
        skel = self.skel
        for pid in range(self.n_person):
            Seq_new[:, pid] = skel.flip_lr(Seq_new[:, pid])
        return MultiPersonSequence(Seq=Seq_new, Mask=Mask_new, skel=skel)

    def get_uid(self) -> str:
        """
        To uniquely identify this sequence we produce a unique id
        ...quite hacky but gets the job done...
        """
        if self.__uid is None:
            uid_txt = f"n_frames:{self.n_frames}, n_person:{self.n_person},"
            uid_txt += f" mask:{np.sum(self.Mask)}"
            Seq_flat, Frames, Pids = nb_get_unmasked_sequences_flat(
                self.Seq, self.Mask
            )  # noqa E501
            Frames = np.array2string(Frames, threshold=len(Frames) + 1)
            Pids = np.array2string(Pids, threshold=len(Frames) + 1)
            uid_txt += f"ff:{Frames}==="
            uid_txt += f"pp:{Pids}"
            uid_txt += sequence_stats_to_unique_str(Seq_flat)
            self.__uid = (
                "mps" + hashlib.sha1(uid_txt.encode("utf-8")).hexdigest()
            )  # noqa B324
        return self.__uid

    def is_masked(self):
        """
        :returns True iff at least one frame has a masked pose
        """
        return np.min(self.Mask) < 0.9

    def n_max_person(self):
        return self.n_person

    def drop_empty_rows(self):
        M = (reduce(self.Mask, "t p -> p", "sum") > 0.5).nonzero()[0]
        if len(M) > 0 and len(M) < self.n_max_person():
            Seq = np.ascontiguousarray(self.Seq[:, M])
            Mask = np.ascontiguousarray(self.Mask[:, M])
            return MultiPersonSequence(Seq=Seq, skel=self.skel, Mask=Mask)
        else:
            return self

    def split_into_sequences_with_equal_n_person(self):
        """
        slits the sequence into subsquences with equal number of
        person each
        """
        all_sequences = []
        last_start = 0
        for t in range(1, self.n_frames):
            if np.sum(np.abs(self.Mask[t] - self.Mask[t - 1])) > 0.01:
                cur_Seq = self.Seq[last_start:t].copy()
                cur_Mask = self.Mask[last_start:t].copy()
                all_sequences.append(
                    MultiPersonSequence(
                        Seq=cur_Seq, Mask=cur_Mask, skel=self.skel
                    ).drop_empty_rows()
                )
                last_start = t
        # handle the last entry
        cur_Seq = self.Seq[last_start : t + 1].copy()  # noqa E501
        cur_Mask = self.Mask[last_start : t + 1].copy()  # noqa E501
        all_sequences.append(
            MultiPersonSequence(
                Seq=cur_Seq, Mask=cur_Mask, skel=self.skel
            ).drop_empty_rows()
        )
        return all_sequences


@nb.njit(
    nb.types.Tuple((nb.float32[:, :], nb.int64[:], nb.int64[:]))(
        nb.float32[:, :, :], nb.float32[:, :]
    ),
    nogil=True,
)
def nb_get_unmasked_sequences_flat(Seq, Mask):
    """
    :param Seq: {n_frames x n_person x jd}
    :param Mask: {n_frames x n_person}
    """
    n_frames = Seq.shape[0]
    n_person = Seq.shape[1]
    jd = Seq.shape[2]
    flat_Seq = np.empty((n_frames * n_person, jd), dtype=np.float32)
    Frames = np.empty((n_frames * n_person), dtype=np.int64)
    Pids = np.empty((n_frames * n_person), dtype=np.int64)

    ptr = 0
    for t in range(n_frames):
        for pid in range(n_person):
            if Mask[t, pid] > 0.5:
                flat_Seq[ptr] = Seq[t, pid]
                Frames[ptr] = t
                Pids[ptr] = pid
                ptr += 1

    return (
        np.ascontiguousarray(flat_Seq[:ptr]),
        np.ascontiguousarray(Frames[:ptr]),
        np.ascontiguousarray(Pids[:ptr]),
    )


def sequence_stats_to_unique_str(seq):
    """
    :param seq: {n_frames x jd}
    """
    if len(seq.shape) != 2:
        raise WeirdShapeError(f"Weird shape: {seq.shape}")
    ma = np.array2string(
        np.round(np.max(seq, axis=0), decimals=2), suppress_small=True
    )  # noqa E501
    mi = np.array2string(
        np.round(np.min(seq, axis=0), decimals=2), suppress_small=True
    )  # noqa E501
    mu = np.array2string(
        np.round(np.mean(seq, axis=0), decimals=2), suppress_small=True
    )
    std = np.array2string(
        np.round(np.std(seq, axis=0), decimals=2), suppress_small=True
    )
    return f"max:{ma}_min:{mi}_mu:{mu}_std:{std}"