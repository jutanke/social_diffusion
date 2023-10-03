import logging
import pickle
import numpy as np
import numpy.linalg as la
import numba as nb
from social_diffusion.skeleton import Skeleton
from social_diffusion.sequence import MultiPersonSequence
from social_diffusion.datasets.panoptic import (
    PANOPTIC_SKELETON,
    interpolate_missing_limbs,
)
from social_diffusion.datasets import frames2segments

from social_diffusion.datasets.dataset import get_LocalMultiPersonDataset
from social_diffusion import get_data_dir
from multiprocessing.pool import ThreadPool
from einops import rearrange, reduce
import torch
from scipy.ndimage import gaussian_filter1d
from os.path import join
import math as m
from typing import List
from time import time
from tqdm import tqdm


logger = logging.getLogger(__name__)


def get_haggling_local_dataset(n_frames: int, n_in: int):
    train_seq = get_haggling_train_sequences()
    test_seq = get_haggling_test_sequences()
    ds = get_LocalMultiPersonDataset(  # noqa E501
        train_seq,
        n_frames=n_frames,
        normalize_at_frame=n_in - 1,
        normalize=False,
        augmentation_random_rotate=False,
        augmentation_random_translate=False,
    )
    ds_test = get_LocalMultiPersonDataset(  # noqa E501
        test_seq,
        n_frames=n_frames,
        normalize_at_frame=n_in - 1,
        normalize=False,
        augmentation_random_rotate=False,
        augmentation_random_translate=False,
    )
    return ds, ds_test, ds.skel


def get_haggling_test_sequences(to_sequence=True) -> List[MultiPersonSequence]:
    """ """
    _start = time()
    TEST = HagglingScene.load_test()
    logger.info(
        "get_haggling_test_sequences => elapsed :%0.4f" % (time() - _start)
    )  # noqa E501
    for scene in tqdm(TEST):
        scene = manually_fix(scene)
        scene.fix_poses()
    if to_sequence:
        return [scene.to_sequence() for scene in TEST]
    else:
        return [scene for scene in TEST]


def get_haggling_train_sequences(to_sequence=True) -> List[MultiPersonSequence]:
    """ """
    _start = time()
    TRAIN = HagglingScene.load_train()
    logger.info(
        "get_haggling_train_sequences => elapsed :%0.4f" % (time() - _start)
    )  # noqa E501
    for scene in tqdm(TRAIN):
        scene = manually_fix(scene)
        scene.fix_poses()
    if to_sequence:
        return [scene.to_sequence() for scene in TRAIN]
    else:
        return [scene for scene in TRAIN]
    # return [scene.to_sequence() for scene in TRAIN]


@nb.njit(nb.float32[:, :, :, :](nb.float32[:, :, :, :]), nogil=True)
def backfill_zeros(seq):
    """
    IF the method starts with zero entries we have to find the first good
    entry!
    :param seq: {n_frames x n_person x J x 4}
    """
    n_frames = seq.shape[0]
    n_person = seq.shape[1]
    n_joints = seq.shape[2]
    for pid in range(n_person):
        for jid in range(n_joints):
            if seq[0, pid, jid, 3] < 0.09:
                for t in range(1, n_frames):
                    if seq[t, pid, jid, 3] > 0.15:
                        for tt in range(0, t):
                            seq[tt, pid, jid, :3] = seq[t, pid, jid, :3]
    return seq


class HagglingScene:
    @staticmethod
    def skeleton() -> Skeleton:
        return PANOPTIC_SKELETON

    @staticmethod
    def load_all():
        global ALL_FILES_NAMES
        Scenes = []
        for scene_name in ALL_FILES_NAMES:
            scene = HagglingScene(scene_name)
            Scenes.append(scene)
        return Scenes

    @staticmethod
    def load_train():
        global ALL_FILES_NAMES
        TRAIN_FILES = [
            scene_name
            for scene_name in ALL_FILES_NAMES
            if scene_name[: scene_name.find("_group")] not in TESTING_SEQUENCES
        ]
        with ThreadPool(len(TRAIN_FILES)) as p:
            return p.map(HagglingScene, TRAIN_FILES)

    @staticmethod
    def load_test():
        global ALL_FILES_NAMES
        Scenes = []
        for scene_name in ALL_FILES_NAMES:
            full_name = scene_name[: scene_name.find("_group")]
            if full_name in TESTING_SEQUENCES:
                scene = HagglingScene(scene_name)
                Scenes.append(scene)
        return Scenes

    def __init__(self, scene_name: str):
        if not scene_name.endswith(".pkl"):
            raise ValueError(f"Name should end on .pkl ({scene_name})")
        self.scene_name = scene_name

        local_path = join(get_data_dir(), "haggling")

        fname_pose = join(
            local_path, f"panopticDB_body_pkl_hagglingProcessed/{scene_name}"
        )  # noqa E501
        fname_speech = join(
            local_path, f"panopticDB_speech_pkl_hagglingProcessed/{scene_name}"
        )  # noqa E501

        with open(fname_pose, "rb") as f:
            P = pickle.load(f, encoding="latin1")
            self.seller_ids = list(sorted(P["sellerIds"]))
            self.left_seller_id = P["leftSellerId"]
            self.right_seller_id = P["rightSellerId"]
            self.buyer_id = P["buyerId"]
            self.winner_id = P["winnerId"]
            self.poses = {}  # pid -> seq
            self.scores = {}  # pid -> scores
            self.normals = {}  # pid -> {"body", "face"}
            for i in range(3):
                J = rearrange(P["subjects"][i]["joints19"], "d t -> t d")
                body_normal = rearrange(
                    P["subjects"][i]["bodyNormal"], "d t -> t d"
                )  # noqa E501
                face_normal = rearrange(
                    P["subjects"][i]["faceNormal"], "d t -> t d"
                )  # noqa E501
                scores = rearrange(P["subjects"][i]["scores"], "d t -> t d")
                # make z point upwards - - -
                z_flip = np.diag([1, 1, -1])
                J = rearrange(J, "t (j d) -> (t j) d", j=19, d=3)
                J[:, [0, 1, 2]] = (J[:, [0, 2, 1]] / 100) @ z_flip
                J = rearrange(J, "(t j) d -> t (j d)", j=19, d=3)

                # we have to flip body/face normals as well!
                body_normal[:, [0, 1, 2]] = (
                    body_normal[:, [0, 2, 1]]
                ) @ z_flip  # noqa E501
                face_normal[:, [0, 1, 2]] = (
                    face_normal[:, [0, 2, 1]]
                ) @ z_flip  # noqa E501
                # - - -

                pid = P["subjects"][i]["humanId"]
                self.poses[pid] = J
                self.scores[pid] = scores

        with open(fname_speech, "rb") as f:
            S = pickle.load(f, encoding="latin1")
            self.speech = {}  # pid -> speech
            self.words = {}  # pid -> words
            for i in range(3):
                pid = S["speechData"][i]["humanId"]
                speech = S["speechData"][i]["indicator"][
                    :-1
                ]  # speech has one entry too many
                self.speech[pid] = speech

        self.n_frames = -1
        for pid in self.poses.keys():
            # assert pid in self.speech
            if pid not in self.speech:
                raise ValueError(f"Cannot find pid:{pid} in speech!")
            if self.n_frames == -1:
                self.n_frames = len(self.poses[pid])
            else:
                if self.n_frames != len(self.poses[pid]):
                    raise ValueError(
                        f"frames and poses inconsistent: {self.n_frames} vs {len(self.poses[pid])}"  # noqa E501
                    )  # noqa E501
            if len(self.speech[pid]) != self.n_frames:
                raise ValueError(
                    f"pid{pid} speech:{self.speech[pid].shape}, poses:{self.poses[pid].shape}"  # noqa E501
                )  # noqa E501

        self.pids = sorted(self.poses.keys())
        self.pid2index = {}
        for index, pid in enumerate(self.pids):
            self.pid2index[pid] = index

    def __len__(self):
        return self.n_frames

    def to_sequence(self) -> MultiPersonSequence:
        return MultiPersonSequence(
            Seq=self.get_poses(), skel=HagglingScene.skeleton()
        )  # noqa E501

    def mean_location(self):
        mu = reduce(
            self.get_poses(), "t p (j d) -> d", reduction="mean", j=19, d=3
        )  # noqa E501
        mu[2] = 0
        return mu

    def which_seller_has_the_attention(
        self, return_speech=False, return_buyer_speech=False
    ):
        if return_buyer_speech:
            if not return_speech:
                raise ValueError("which_seller_has_the_attention fail")
        HO = self.calculate_head_orientations()

        pid2idx = {}
        for i, pid in enumerate(self.pids):
            pid2idx[pid] = i

        seller1 = HO[:, pid2idx[self.seller_ids[0]]]
        seller2 = HO[:, pid2idx[self.seller_ids[1]]]
        buyer = HO[:, pid2idx[self.buyer_id]]

        seller1_loc = seller1[:, 0]
        seller2_loc = seller2[:, 0]
        buyer_loc = buyer[:, 0]
        buyer_lookdir = buyer[:, 1]

        def get_dir_vec(a, b):
            ab = b - a
            return ab / la.norm(ab, axis=1, keepdims=True)

        b_to_s1 = get_dir_vec(buyer_loc, seller1_loc)
        b_to_s2 = get_dir_vec(buyer_loc, seller2_loc)

        att1 = np.einsum("ij,ij->i", b_to_s1, buyer_lookdir)
        att2 = np.einsum("ij,ij->i", b_to_s2, buyer_lookdir)

        attn = np.zeros((self.n_frames, 2), dtype=np.int64)
        for t in range(self.n_frames):
            if att1[t] > att2[t]:
                attn[t, 0] = 1
            else:
                attn[t, 1] = 1

        if return_speech:
            all_speech = self.get_speech()  # n_frames x n_person
            speech1 = rearrange(
                all_speech[:, pid2idx[self.seller_ids[0]]], "t -> t 1"
            )  # noqa E501
            speech2 = rearrange(
                all_speech[:, pid2idx[self.seller_ids[1]]], "t -> t 1"
            )  # noqa E501
            speech = np.concatenate([speech1, speech2], axis=1)

            if return_buyer_speech:
                buyer_speech = all_speech[:, pid2idx[self.buyer_id]]
                return attn, speech, buyer_speech
            else:
                return attn, speech
        else:
            return attn

    def resize(self, start: int, end: int):
        for pid in self.pids:
            self.poses[pid] = self.poses[pid][start:end]
            self.speech[pid] = self.speech[pid][start:end]
            self.scores[pid] = self.scores[pid][start:end]
        self.n_frames = len(self.poses[pid])

    def fix_poses(self):
        n_person = 3
        m = np.ones((self.n_frames), dtype=np.int64)
        Seq = self.get_poses_with_scores()

        # step 1: backfill
        # Seq = backfill_zeros(Seq)

        with ThreadPool(n_person) as p:
            PM_new = p.starmap(
                interpolate_missing_limbs,
                [(Seq[:, i], m, i) for i in range(n_person)],  # noqa E501
            )
            P_new = np.array([P for P, _ in PM_new], dtype=np.float32)

        for i, pid in enumerate(self.pids):
            seq = P_new[i]  # t x 19 x 4
            seq = seq[:, :, :3]
            self.poses[pid] = rearrange(seq, "t j d -> t (j d)")

    def get_poses_with_scores(self, sigma=-1.0):
        poses = rearrange(
            self.get_poses(sigma=sigma), "t p (j d) -> t p j d", j=19, d=3
        )
        scores = rearrange(self.get_scores(), "t p j -> t p j 1")
        return np.concatenate([poses, scores], axis=3)

    def get_scores(self):
        scores = np.empty((self.n_frames, 3, 19), dtype=np.float32)
        for i, pid in enumerate(self.pids):
            scores[:, i] = self.scores[pid]
        return scores

    def get_poses(self, sigma=0.5):
        poses = np.empty((self.n_frames, 3, 57), dtype=np.float32)
        for i, pid in enumerate(self.pids):
            poses[:, i] = self.poses[pid]
        if sigma > 0:
            poses = smooth_poses(poses, sigma=sigma)
        return poses

    def get_ordered_poses(self, sigma=0.5):
        P = self.get_poses(sigma=sigma)

        buyer_index = self.pid2index[self.buyer_id]
        left_index = self.pid2index[self.left_seller_id]
        right_index = self.pid2index[self.right_seller_id]

        return np.ascontiguousarray(
            P[:, [buyer_index, left_index, right_index]]
        )  # noqa E501

    def update_poses(self, poses: np.ndarray, scores: np.ndarray = None):
        assert len(poses) == self.n_frames
        for idx in range(3):
            pid = self.pids[idx]
            self.poses[pid] = poses[:, idx]
        if scores is not None:
            assert len(scores) == self.n_frames
            for idx in range(3):
                pid = self.pids[idx]
                self.scores[pid] = scores[:, idx]

    def get_plot_description(self):
        """
        return:
            per_person_text: {
                pid, t --> "buyer|<talk>"
            }
        """
        per_person_text = {}
        for t in range(self.n_frames):
            for i, pid in enumerate(self.pids):
                txt = ""
                if pid == self.buyer_id:
                    txt += "<buyer>"
                elif pid == self.left_seller_id:
                    txt += "<left seller>"
                elif pid == self.right_seller_id:
                    txt += "<right seller>"
                else:
                    raise ValueError(
                        f"Cannot assign {pid} for {self.scene_name}"
                    )  # noqa E501
                if self.speech[pid][t] > 0.5:
                    txt += "|<speak>"
                per_person_text[i, t] = txt
        return per_person_text

    def get_speech(self):
        speech = np.empty((self.n_frames, 3), dtype=np.int64)
        for i, pid in enumerate(self.pids):
            speech[:, i] = self.speech[pid]
        return speech

    def calculate_head_orientations(self):
        """
        Our OWN head orientation calculation
        return:
            {n_frames x n_person x 2 x 2}  # (pos / norm)
        """
        n_person = 3
        poses = self.get_poses()
        head_orientations = np.empty(
            (self.n_frames, n_person, 2, 2), dtype=np.float32
        )  # noqa E501
        for i in range(n_person):
            seq = poses[:, i]
            head_orientations[:, i, 0] = rearrange(
                seq, "t (j d) -> t j d", j=19, d=3
            )[  # noqa E501
                :, 1, :2
            ]
            head_orientations[:, i, 1] = get_normal(
                seq, left_jid=15, right_jid=17
            )  # noqa E501
        return head_orientations

    def calculate_body_orientations(self):
        n_person = 3
        poses = self.get_poses()
        body_orientations = np.empty(
            (self.n_frames, n_person, 2, 2), dtype=np.float32
        )  # noqa E501
        for i in range(n_person):
            seq = poses[:, i]
            body_orientations[:, i, 0] = rearrange(
                seq, "t (j d) -> t j d", j=19, d=3
            )[  # noqa E501
                :, 2, :2
            ]
            body_orientations[:, i, 1] = get_normal(
                seq, left_jid=6, right_jid=12
            )  # noqa E501
        return body_orientations

    def has_problematic_lower_parts(self):
        txt = "- - - - - - - - - - - - - - - - - - "
        P = self.get_poses_with_scores()
        something_is_broken = False
        for pid in range(3):
            pp = P[:, pid, [6, 7, 8, 12, 13, 14], 3]
            broken_lp_indices = ((np.min(pp, axis=1) < 0) * 1).nonzero()[0]
            if len(broken_lp_indices) > 0:
                for start, end, length in frames2segments(
                    broken_lp_indices, include_length=True
                ):
                    if length > 10:
                        something_is_broken = True
                        txt += f"\npid{pid} -> {start} to {end} @{self.scene_name}"  # noqa E501
        if something_is_broken:
            print(txt)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# U T I L S
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def rot2d(theta):
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )


def get_normal(poses, left_jid: int, right_jid: int, smooth=True):
    """
    :param poses: {n_frames x 57}
    """
    is_torch = True
    if isinstance(poses, np.ndarray):
        is_torch = False
        poses = torch.from_numpy(poses)
    poses = rearrange(poses, "t (j d) -> t j d", j=19, d=3)
    left2d = poses[:, left_jid, :2]
    right2d = poses[:, right_jid, :2]
    y = right2d - left2d  # n_frames x 2
    y = y / (torch.norm(y, dim=1, p=2).unsqueeze(1) + 0.00000001)
    if not is_torch:
        y = y.numpy()
    # y[:, [0, 1]] = y[:, [1, 0]]

    R = rot2d(-m.pi / 2)

    y = y @ R

    if smooth:
        sigma = 5.0
        y[:, 0] = gaussian_filter1d(y[:, 0], sigma=sigma)
        y[:, 1] = gaussian_filter1d(y[:, 1], sigma=sigma)

    y = y / la.norm(y, axis=1, keepdims=True)

    return y


def smooth_poses(poses, sigma, sigma_hip=1.5):
    """
    :param poses: {n_frames x n_person x 56}
    """
    hip_dims = set(
        [
            2 * 3,
            2 * 3 + 1,
            2 * 3 + 2,
            6 * 3,
            6 * 3 + 1,
            6 * 3 + 2,
            12 * 3,
            12 * 3 + 1,
            12 * 3 + 2,
        ]
    )

    n_person = poses.shape[1]
    n_dim = poses.shape[2]
    assert len(poses.shape) == 3
    for pid in range(n_person):
        for dim in range(n_dim):
            if dim in hip_dims:
                _sigma = sigma_hip
            else:
                _sigma = sigma
            poses[:, pid, dim] = gaussian_filter1d(
                poses[:, pid, dim], sigma=_sigma
            )  # noqa E501
    return poses


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ===== M A N U A L L Y  F I X =====
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TESTING_SEQUENCES = set(
    [  # https://github.com/CMU-Perceptual-Computing-Lab/ssp
        "170221_haggling_b1",
        "170221_haggling_b2",
        "170221_haggling_b3",
        "170228_haggling_b1",
        "170228_haggling_b2",
        "170228_haggling_b3",
    ]
)

ALL_FILES_NAMES = [
    "170221_haggling_b1_group0.pkl",
    "170221_haggling_b1_group1.pkl",
    "170221_haggling_b1_group2.pkl",
    "170221_haggling_b1_group3.pkl",
    "170221_haggling_b1_group4.pkl",
    "170221_haggling_b1_group5.pkl",
    "170221_haggling_b2_group0.pkl",
    "170221_haggling_b2_group1.pkl",
    "170221_haggling_b2_group2.pkl",
    "170221_haggling_b2_group3.pkl",
    "170221_haggling_b2_group4.pkl",
    "170221_haggling_b2_group5.pkl",
    "170221_haggling_b3_group0.pkl",
    "170221_haggling_b3_group1.pkl",
    "170221_haggling_b3_group2.pkl",
    "170221_haggling_m1_group0.pkl",
    "170221_haggling_m1_group1.pkl",
    "170221_haggling_m1_group2.pkl",
    "170221_haggling_m1_group3.pkl",
    "170221_haggling_m1_group4.pkl",
    "170221_haggling_m1_group5.pkl",
    "170221_haggling_m2_group0.pkl",
    "170221_haggling_m2_group1.pkl",
    "170221_haggling_m2_group2.pkl",
    "170221_haggling_m2_group3.pkl",
    "170221_haggling_m2_group4.pkl",
    "170221_haggling_m2_group5.pkl",
    "170221_haggling_m3_group0.pkl",
    "170221_haggling_m3_group1.pkl",
    "170221_haggling_m3_group2.pkl",
    "170224_haggling_a1_group0.pkl",
    "170224_haggling_a1_group1.pkl",
    # "170224_haggling_a1_group2.pkl",  # broken lower body parts
    "170224_haggling_a1_group3.pkl",
    "170224_haggling_a1_group4.pkl",
    "170224_haggling_a1_group5.pkl",
    "170224_haggling_a1_group6.pkl",
    # "170224_haggling_a1_group7.pkl",  # broken lower body parts
    "170224_haggling_a2_group0.pkl",
    "170224_haggling_a2_group1.pkl",
    "170224_haggling_a2_group2.pkl",
    "170224_haggling_a2_group3.pkl",
    # "170224_haggling_a2_group4.pkl",  # broken lower body parts
    "170224_haggling_a2_group5.pkl",
    "170224_haggling_a2_group6.pkl",
    "170224_haggling_a2_group7.pkl",
    "170224_haggling_a3_group0.pkl",
    # "170224_haggling_a3_group1.pkl",  # broken lower body parts
    # "170224_haggling_a3_group2.pkl",  # broken lower body parts
    "170224_haggling_a3_group3.pkl",
    "170224_haggling_b1_group0.pkl",
    "170224_haggling_b1_group1.pkl",
    "170224_haggling_b1_group2.pkl",
    "170224_haggling_b1_group3.pkl",
    "170224_haggling_b1_group4.pkl",
    "170224_haggling_b1_group5.pkl",
    "170224_haggling_b1_group6.pkl",
    "170224_haggling_b1_group7.pkl",
    "170224_haggling_b2_group0.pkl",
    "170224_haggling_b2_group1.pkl",
    "170224_haggling_b2_group2.pkl",
    "170224_haggling_b2_group3.pkl",
    "170224_haggling_b2_group4.pkl",
    "170224_haggling_b2_group5.pkl",
    "170224_haggling_b2_group6.pkl",
    "170224_haggling_b2_group7.pkl",
    "170224_haggling_b3_group0.pkl",
    "170224_haggling_b3_group1.pkl",
    "170224_haggling_b3_group2.pkl",
    "170224_haggling_b3_group3.pkl",
    "170228_haggling_a1_group0.pkl",
    "170228_haggling_a1_group1.pkl",
    "170228_haggling_a1_group2.pkl",
    "170228_haggling_a1_group3.pkl",
    "170228_haggling_a1_group4.pkl",
    "170228_haggling_a1_group5.pkl",
    "170228_haggling_a1_group6.pkl",
    "170228_haggling_a1_group7.pkl",
    "170228_haggling_a2_group0.pkl",
    "170228_haggling_a2_group1.pkl",
    "170228_haggling_a2_group2.pkl",
    # "170228_haggling_a2_group3.pkl",  # BROKEN
    "170228_haggling_a2_group4.pkl",
    "170228_haggling_a2_group5.pkl",
    "170228_haggling_a2_group6.pkl",
    "170228_haggling_a2_group7.pkl",
    "170228_haggling_a3_group0.pkl",
    "170228_haggling_a3_group1.pkl",
    "170228_haggling_a3_group2.pkl",
    "170228_haggling_a3_group3.pkl",
    "170228_haggling_b1_group0.pkl",
    "170228_haggling_b1_group1.pkl",
    "170228_haggling_b1_group2.pkl",
    "170228_haggling_b1_group3.pkl",
    "170228_haggling_b1_group4.pkl",
    "170228_haggling_b1_group5.pkl",
    "170228_haggling_b1_group6.pkl",
    "170228_haggling_b1_group7.pkl",
    "170228_haggling_b1_group8.pkl",
    "170228_haggling_b1_group9.pkl",
    "170228_haggling_b2_group0.pkl",
    "170228_haggling_b2_group1.pkl",
    "170228_haggling_b2_group2.pkl",
    "170228_haggling_b2_group3.pkl",
    "170228_haggling_b2_group4.pkl",
    "170228_haggling_b2_group5.pkl",
    "170228_haggling_b2_group6.pkl",
    "170228_haggling_b2_group7.pkl",
    "170228_haggling_b2_group8.pkl",
    "170228_haggling_b2_group9.pkl",
    "170228_haggling_b3_group0.pkl",
    "170228_haggling_b3_group1.pkl",
    "170228_haggling_b3_group2.pkl",
    "170228_haggling_b3_group3.pkl",
    "170228_haggling_b3_group4.pkl",
    "170404_haggling_a1_group0.pkl",
    "170404_haggling_a1_group1.pkl",
    "170404_haggling_a1_group2.pkl",
    "170404_haggling_a1_group3.pkl",
    "170404_haggling_a2_group0.pkl",
    "170404_haggling_a2_group1.pkl",
    "170404_haggling_a2_group2.pkl",
    "170404_haggling_a2_group3.pkl",
    "170404_haggling_a3_group0.pkl",
    "170404_haggling_a3_group1.pkl",
    "170404_haggling_b1_group0.pkl",
    "170404_haggling_b1_group1.pkl",
    "170404_haggling_b1_group2.pkl",
    "170404_haggling_b1_group3.pkl",
    "170404_haggling_b1_group4.pkl",
    "170404_haggling_b1_group5.pkl",
    "170404_haggling_b1_group6.pkl",
    "170404_haggling_b1_group7.pkl",
    "170404_haggling_b2_group0.pkl",
    "170404_haggling_b2_group1.pkl",
    "170404_haggling_b2_group2.pkl",
    "170404_haggling_b2_group3.pkl",
    "170404_haggling_b2_group4.pkl",
    "170404_haggling_b2_group5.pkl",
    "170404_haggling_b2_group6.pkl",
    "170404_haggling_b2_group7.pkl",
    "170404_haggling_b3_group0.pkl",
    "170404_haggling_b3_group1.pkl",
    "170404_haggling_b3_group2.pkl",
    "170404_haggling_b3_group3.pkl",
    # "170407_haggling_a1_group0.pkl",  # BROKEN
    "170407_haggling_a1_group1.pkl",
    "170407_haggling_a1_group2.pkl",
    "170407_haggling_a1_group3.pkl",
    "170407_haggling_a1_group4.pkl",
    "170407_haggling_a1_group5.pkl",
    "170407_haggling_a2_group0.pkl",
    "170407_haggling_a2_group1.pkl",
    "170407_haggling_a2_group2.pkl",
    "170407_haggling_a2_group3.pkl",
    "170407_haggling_a2_group4.pkl",
    "170407_haggling_a2_group5.pkl",
    "170407_haggling_a3_group0.pkl",
    "170407_haggling_a3_group1.pkl",
    "170407_haggling_a3_group2.pkl",
    "170407_haggling_b1_group0.pkl",
    "170407_haggling_b1_group1.pkl",
    "170407_haggling_b1_group2.pkl",
    "170407_haggling_b1_group3.pkl",
    "170407_haggling_b1_group4.pkl",
    "170407_haggling_b1_group5.pkl",
    "170407_haggling_b1_group6.pkl",
    "170407_haggling_b1_group7.pkl",
    "170407_haggling_b2_group0.pkl",
    "170407_haggling_b2_group1.pkl",
    "170407_haggling_b2_group2.pkl",
    "170407_haggling_b2_group3.pkl",
    "170407_haggling_b2_group4.pkl",
    "170407_haggling_b2_group5.pkl",
    "170407_haggling_b2_group6.pkl",
    "170407_haggling_b2_group7.pkl",
    "170407_haggling_b3_group0.pkl",
    "170407_haggling_b3_group1.pkl",
    "170407_haggling_b3_group2.pkl",
    "170407_haggling_b3_group3.pkl",
]


def manually_fix(scene: HagglingScene):
    """
    :param sequence:  {Scene}
    """
    global MANUAL_FIX_LEGS, MANUAL_FIX, MANUAL_RESIZE

    # --- GENERAL BODY ---
    if scene.scene_name in MANUAL_FIX:
        for entry in MANUAL_FIX[scene.scene_name]:
            pid = entry["pid"]  # not actual pid but index into poses
            start = entry["start"]
            end = entry["end"]
            jids = entry["jids"]

            poses = rearrange(
                scene.get_poses(), "t p (j d) -> t p j d", j=19, d=3
            )  # noqa E501
            scores = scene.get_scores()  # t p j
            for jid in jids:
                left = poses[start, pid, jid]
                right = poses[end, pid, jid]
                interp = np.linspace(left, right, num=end - start)
                for idx, t in enumerate(range(start + 1, end)):
                    tar = interp[idx]
                    cur = poses[t, pid, jid]
                    if la.norm(cur - tar) > 0.1:
                        poses[t, pid, jid] = tar
                        scores[t, pid, jid] = 2.0
            scene.update_poses(
                rearrange(poses, "t p j d -> t p (j d)"), scores
            )  # noqa E501

    # --- LEGS ---
    LEG_JIDS = [2, 6, 7, 8, 12, 13, 14]
    LEG_JID_INDICES = []
    for jid in LEG_JIDS:
        LEG_JID_INDICES.append(jid * 3)
        LEG_JID_INDICES.append(jid * 3 + 1)
        LEG_JID_INDICES.append(jid * 3 + 2)

    if scene.scene_name in MANUAL_FIX_LEGS:
        poses = scene.get_poses()  # n_frames x n_person x 57
        scores = scene.get_scores()  # t p j
        for entry in MANUAL_FIX_LEGS[scene.scene_name]:
            pid = entry["pid"]  # not actual pid but index into poses
            good_frames = entry["good_frames"]

            # step 0 - pre-fill to 0
            # TODO

            # step 1 - fill-up in the middle
            for i in range(len(good_frames) - 1):
                frame_a = good_frames[i]
                frame_b = good_frames[i + 1]
                if frame_a >= frame_b:  # assert frame_a < frame_b
                    raise ValueError(f"{frame_a} vs {frame_b}")
                pose_a = poses[frame_a, pid]
                pose_b = poses[frame_b, pid]
                interp = np.linspace(pose_a, pose_b, num=frame_b - frame_a)
                for j, t in enumerate(range(frame_a + 1, frame_b)):
                    for idx in LEG_JID_INDICES:
                        tar = interp[j, idx]
                        cur = poses[t, pid, idx]
                        if abs(cur - tar) > 0.1:
                            poses[t, pid, idx] = tar
                            scores[t, pid, jid] = 2.0

            # step 3 - fill-up to end
            # TODO

            # step 4 - make sure that <2> is ~ btwn <6> & <12>
            poses = rearrange(poses, "t p (j d) -> t p j d", j=19, d=3)
            n_frames = len(poses)
            for t in range(0, n_frames):
                center = poses[t, pid, 2]
                left = poses[t, pid, 6]
                right = poses[t, pid, 12]
                supposed_center = (left + right) / 2
                d = la.norm(supposed_center - center)
                if d > 0.01:
                    poses[t, pid, 2] = supposed_center
                    scores[t, pid, 2] = 2.0
            poses = rearrange(poses, "t p j d -> t p (j d)")

        # scene.poses = poses
        scene.update_poses(poses, scores)

    if scene.scene_name in MANUAL_RESIZE:
        entry = MANUAL_RESIZE[scene.scene_name]
        start = 0
        end = len(scene)
        if "start" in entry:
            start = entry["start"]
        if "end" in entry:
            end = entry["end"]
        scene.resize(start, end)

    return scene


MANUAL_RESIZE = {
    "170228_haggling_a1_group5.pkl": {"start": 38},
    "170407_haggling_a2_group4.pkl": {"end": 1822},
}


MANUAL_FIX = {
    "170224_haggling_a2_group7.pkl": [
        {
            "pid": 2,
            "start": 1587,
            "end": 1603,
            "jids": [0, 1, 3, 4, 5, 9, 10, 11, 15, 16, 17, 18],
        }
    ],
    "170228_haggling_a1_group5.pkl": [
        {"pid": 1, "start": 714, "end": 722, "jids": [13, 14]},
        {"pid": 1, "start": 722, "end": 750, "jids": [7, 8, 13, 14]},
    ],
    "170407_haggling_a2_group4.pkl": [
        {"pid": 1, "start": 63, "end": 69, "jids": [10, 11]},
        {"pid": 1, "start": 1531, "end": 1669, "jids": [12, 13, 14]},
    ],
}

MANUAL_FIX_LEGS = {
    "170407_haggling_a2_group4.pkl": [
        {
            "pid": 1,
            "good_frames": [
                0,
                4,
                9,
                14,
                19,
                29,
                37,
                44,
                49,
                53,
                62,
                66,
                71,
                75,
                85,
                89,
                94,
                100,
                108,
                116,
                125,
                133,
                138,
                141,
                148,
                177,
                270,
                380,
                456,
                471,
                476,
                490,
                586,
                616,
                702,
                706,
                710,
                729,
                742,
                757,
                772,
                821,
                905,
                930,
                966,
                1002,
                1051,
                1101,
                1115,
                1139,
                1152,
                1175,
                1237,
                1312,
                1403,
                1505,
                1527,
                1664,
                1693,
                1752,
                1773,
                1819,
            ],
        },
    ],
    "170224_haggling_a2_group7.pkl": [
        {
            "pid": 2,
            "good_frames": [
                0,
                5,
                10,
                15,
                18,
                20,
                24,
                26,
                30,
                32,
                34,
                36,
                39,
                41,
                46,
                53,
                58,
                61,
                64,
                67,
                71,
                74,
                76,
                79,
                81,
                87,
                91,
                95,
                102,
                114,
                121,
                173,
                212,
                247,
                251,
                256,
                261,
                273,
                279,
                283,
                290,
                296,
                306,
                444,
                488,
                602,
                758,
                884,
                1128,
                1346,
                1430,
                1435,
                1444,
                1449,
                1453,
                1460,
                1469,
                1543,
                1577,
                1613,
                1618,
                1629,
                1636,
                1641,
                1648,
                1653,
                1661,
                1668,
                1674,
                1681,
                1685,
                1688,
                1694,
                1698,
                1704,
                1710,
                1717,
                1740,
                1777,
                1819,
            ],
        }
    ],
    "170407_haggling_a1_group4.pkl": [
        {
            "pid": 2,
            "good_frames": [
                13,
                38,
                61,
                95,
                152,
                257,
                572,
                796,
                1064,
                1385,
                1810,
            ],  # noqa E501
        }
    ],
    "170407_haggling_a2_group0.pkl": [
        {
            "pid": 1,
            "good_frames": [
                0,
                22,
                31,
                33,
                38,
                46,
                49,
                90,
                99,
                109,
                136,
                194,
                241,
                349,
                386,
                488,
                519,
                549,
                657,
                681,
                743,
                785,
                815,
                865,
                935,
                1029,
                1389,
                1767,
                1785,
            ],
        }
    ],
}