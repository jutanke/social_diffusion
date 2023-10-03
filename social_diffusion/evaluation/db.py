import hashlib
import math as m

from os.path import isfile
from typing import List

import numba as nb
import numpy as np

import numpy.linalg as la

# import torch
from annoy import AnnoyIndex

from social_diffusion.transforms.transforms import normalize

from einops import rearrange, reduce
from tqdm import tqdm


def compress_seq(seq, relevant_jids):
    """
    :param seq: {n_frames x 56}
    """
    seq = rearrange(seq.copy(), "t (j d) -> t j d", j=19, d=3)
    seq = normalize(seq, frame=0, jid_left=6, jid_right=12, fix_shape=False)
    return rearrange(seq[:, relevant_jids], "t j d -> (t j d)")


class Database:
    def __init__(self, scenes: List, motion_word_size=10, ss=1):
        self.relevant_jids = [1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14]

        self.compress_fn = compress_seq

        self.motion_word_size = motion_word_size
        assert int(m.ceil(motion_word_size / ss)) == int(
            m.floor(motion_word_size / ss)
        ), f"{m.ceil(motion_word_size / ss)} vs {int(m.floor(motion_word_size / ss))}"  # noqa E501
        n_dim = (len(self.relevant_jids) * 3) * motion_word_size // ss
        self.ss = ss
        self.meta = []
        self.lookup = AnnoyIndex(n_dim, "euclidean")

        # check if we have the db already!
        unique_txt = f"rel_ids: {self.relevant_jids} motion_word_size:{motion_word_size}, ss:{ss} ~ ~"  # noqa E501
        scene_names = sorted([scene.scene_name for scene in scenes])
        unique_txt += "++".join(scene_names)
        hash_object = hashlib.sha256(unique_txt.encode("utf-8"))
        unique_fname = f"/tmp/db_{hash_object.hexdigest()}.ann"

        if isfile(unique_fname):
            self.lookup.load(unique_fname)
        else:
            for scene in tqdm(scenes):
                poses = scene.get_poses()
                n_frames = poses.shape[0]
                n_person = poses.shape[1]
                speech = scene.get_speech()

                for t in range(n_frames - motion_word_size):
                    for i in range(n_person):
                        seq = poses[t : t + motion_word_size, i]  # noqa E501
                        speech_seq = speech[t : t + motion_word_size, i]  # noqa E501
                        try:
                            word = self.compress_fn(
                                seq[::ss], self.relevant_jids
                            )  # noqa E501
                            self.lookup.add_item(len(self.meta), word)
                            self.meta.append(
                                {
                                    "pid": i,
                                    "t": t,
                                    "name": scene.scene_name,
                                    "speech": speech_seq,
                                }
                            )
                        except:  # noqa E722
                            seq = rearrange(seq, "t (j d) -> t j d", j=19, d=3)
                            print(
                                f"\nFAILED AT t:{t} for pid{i} @{scene.scene_name}"  # noqa E501
                            )  # noqa E501
                            print("\tz-left:", seq[:, 6, 2])
                            print("\tz-right:", seq[:, 12, 2])
                            raise
            self.lookup.build(-1)
            self.lookup.save(unique_fname)

    def query(self, subseq):
        """
        :param subseq: {n_dim}
        """
        i = self.lookup.get_nns_by_vector(
            vector=subseq, n=1, include_distances=False
        )  # noqa E501
        i = i[0]
        true_seq = self.lookup.get_item_vector(i)
        kernel_size = self.motion_word_size
        true_seq = np.reshape(true_seq, (kernel_size, -1))
        query_seq = np.reshape(subseq, (kernel_size, -1))
        dist = ndms(true_seq, query_seq, kernel_size=kernel_size)
        return dist, i

    def rolling_query(self, subseq):
        """
        :param subseq: {n_frames x dim}
        """
        subseq = subseq.astype("float64")
        kernel_size = self.motion_word_size
        assert len(subseq) > kernel_size, str(subseq.shape)
        distances = []
        identities = []
        for frame in range((len(subseq) - kernel_size) + 1):
            sub_seq = subseq[frame : frame + kernel_size].copy()  # noqa E203
            word = self.compress_fn(sub_seq[:: self.ss], self.relevant_jids)
            word = word.flatten()
            d, i = self.query(word)
            distances.append(d)
            identities.append(i)
        distances = np.array(distances)
        return distances, identities

    def speech_inference(self, seq):
        """
        :param seq: {n_frames x 54}
        """
        n_frames = len(seq)
        assert n_frames >= self.motion_word_size

        speech = []
        for t in range(n_frames - self.motion_word_size):
            subseq = seq[t : t + self.motion_word_size]  # noqa E501
            try:
                word = self.compress_fn(subseq[:: self.ss], self.relevant_jids)
            except:  # noqa E722
                word = compress_seq(subseq, self.relevant_jids)
            i = self.lookup.get_nns_by_vector(
                vector=word, n=1, include_distances=False
            )[0]
            speech.append(np.mean(self.meta[i]["speech"]))
        speech = np.array(speech)
        return speech


@nb.njit(nb.float64[:, :](nb.float64[:, :]), nogil=True)
def velocity(seq):
    """
    :param [n_frames x dim]
    """
    n_frames = len(seq)
    dim = seq.shape[1]
    V = np.empty((n_frames - 1, dim), dtype=np.float64)
    for t in range(n_frames - 1):
        a = seq[t]
        b = seq[t + 1]
        v = b - a
        V[t] = v
    return V


@nb.njit(nb.float64(nb.float64[:, :], nb.float64[:, :], nb.int64), nogil=True)
def ndms(true_seq, query_seq, kernel_size):
    """
    :param true_seq: [kernel_size x n_useful_joints*3]
    :param query_seq: [kernel_size x n_useful_joints*3]
    """
    eps = 0.0000001
    true_v_ = np.ascontiguousarray(velocity(true_seq))
    query_v_ = np.ascontiguousarray(velocity(query_seq))
    dim = true_v_.shape[1]
    n_features = dim // 3
    true_v = true_v_.reshape((kernel_size - 1, n_features, 3))
    query_v = query_v_.reshape((kernel_size - 1, n_features, 3))
    total_score = 0.0
    for t in range(kernel_size - 1):
        for jid in range(n_features):
            a = np.expand_dims(true_v[t, jid], axis=0)
            b_T = np.expand_dims(query_v[t, jid], axis=1)
            norm_a = max(la.norm(a), eps)
            norm_b = max(la.norm(b_T), eps)
            cos_sim = (a @ b_T) / (norm_a * norm_b)
            disp = min(norm_a, norm_b) / max(norm_a, norm_b)
            score = ((cos_sim + 1) * disp) / 2.0
            total_score += score[0, 0] / n_features
    return total_score / (kernel_size - 1)