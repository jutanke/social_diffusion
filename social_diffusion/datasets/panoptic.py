import numpy as np
import numba as nb
import numpy.linalg as la
from social_diffusion.skeleton import Skeleton
from social_diffusion.sequence import MultiPersonSequence
from social_diffusion.datasets import load_json, frames2segments
from social_diffusion import get_cache_dir, get_data_dir
from multiprocessing.pool import ThreadPool
from json import JSONDecodeError
from einops import rearrange, reduce
from os.path import isfile, isdir, join
from os import listdir

from typing import List
import logging
from time import time


logger = logging.getLogger(__name__)


PANOPTIC_SKELETON = Skeleton(
    normalizing_left_jid=6,
    normalizing_right_jid=12,
    n_joints=19,
    skeleton=[
        (0, 3),
        (3, 4),
        (4, 5),
        (0, 9),
        (9, 10),
        (10, 11),
        (0, 1),
        (0, 2),
        (2, 6),
        (2, 12),
        (6, 7),
        (7, 8),
        (12, 13),
        (13, 14),
        (15, 16),
        (15, 17),
        (17, 18),
        (1, 17),
        (1, 15),
    ],
    left_jids=[15, 16, 3, 4, 5, 6, 7, 8],
    right_jids=[17, 18, 9, 10, 11, 12, 13, 14],
    leg_joints=[7, 8, 13, 14],
)


class PanopticScene:
    """
    Contains a fixed amount of persons in the scene
    """

    def __init__(self, P: np.ndarray, name: str, skel: Skeleton):
        """
        :param P: {n_frames x n_person x jd}
        """
        self.Seq = rearrange(P, "t p j d -> t p (j d)")
        self.name = name
        self.skel = skel

    def n_person(self):
        return self.Seq.shape[1]

    def __len__(self):
        return len(self.Seq)

    def to_sequence(self) -> MultiPersonSequence:
        return MultiPersonSequence(Seq=self.Seq, skel=PANOPTIC_SKELETON)

    def trim(self, trim: int):
        """
        :param trim: cuts off at the start/end
        """
        if trim == 0:
            return self
        if trim > 0 and (2 * trim) < len(self.Seq):
            Seq = self.Seq[trim:-trim]
            name = self.name + f"_trim{trim}"
            skel = self.skel
            return PanopticScene(
                P=rearrange(Seq, "t p (j d) -> t p j d", d=3).copy(),
                name=name,
                skel=skel,
            )
        else:
            raise ValueError(
                f"Trim cannot be applied to the sequence: trim:{trim} vs Seq:{self.Seq.shape}"  # noqa E501
            )


class PanopticSequence:
    """
    Continous sequence of person(s)
    """

    def __init__(self, P: np.ndarray, M: np.ndarray, name: str):
        """
        :param P: {n_persons x n_persons x 19 x 3}
        :param M: {n_persons x n_persons}
        """
        self.name = name
        self.n_persons = P.shape[1]
        self.n_frames = P.shape[0]
        self.P = P
        self.M = M

    def __lt__(self, other):
        return len(self) < len(other)

    def __len__(self):
        return self.n_frames

    def count_persons(self):
        return self.n_persons

    def split_into_scenes_with_same_amount_of_persons(
        self,
    ) -> List[PanopticScene]:  # noqa E501
        """ """
        count = reduce(self.M, "t p -> t", "sum")
        n_max_person = self.M.shape[1]
        Seqs = []
        cur_P = []
        current_pointer = 0
        for t in range(len(count) - 1):
            if np.sum(np.abs(self.M[t] - self.M[t + 1])) > 0.1:
                cur_P = np.array(cur_P).copy()  # make sure to copy
                valid_pids = []
                for pid in range(n_max_person):
                    if self.M[t, pid] > 0.5:
                        valid_pids.append(pid)
                if len(valid_pids) == 0:
                    print("cnt", count[t], n_max_person)
                    print("self.M[t", self.M[t])
                    print("wtf?", self.name, t)
                # assert len(valid_pids) > 0
                if len(valid_pids) == 0:
                    raise ValueError("No persons..")
                valid_pids = np.array(valid_pids)
                P = np.ascontiguousarray(cur_P[:, valid_pids])
                # M = np.ones((P.shape[0], P.shape[1]), dtype=np.int64)
                Seqs.append(
                    PanopticScene(
                        P=P,
                        name=self.name + f"|esaop{current_pointer}",
                        skel=PANOPTIC_SKELETON,
                    )
                )
                current_pointer += 1
                cur_P = []
            else:
                cur_P.append(self.P[t])
        # handle last one!
        cur_P = np.array(cur_P).copy()  # make sure to copy
        valid_pids = []
        for pid in range(n_max_person):
            if self.M[t, pid] > 0.5:
                valid_pids.append(pid)
        if len(valid_pids) == 0:
            print("wtf?", self.name, t)
        if len(valid_pids) == 0:
            raise ValueError("No valid pids")

        valid_pids = np.array(valid_pids)
        P = np.ascontiguousarray(cur_P[:, valid_pids])
        Seqs.append(
            PanopticScene(
                P=P,
                name=self.name + f"|esaop{current_pointer}",
                skel=PANOPTIC_SKELETON,
            )
        )

        return Seqs


def load_poses3d(json_fname):
    """ """
    try:
        frame = int(
            json_fname.split("/")[-1].replace("body3DScene_", "").replace(".json", "")
        )
    except ValueError as e:
        logger.error(f"it seems the naming has been off: {json_fname}")
        raise e
    if not isfile(json_fname):
        raise ValueError(f"Cannot find {json_fname} :(")
    bodies = {}  # pid:
    try:
        data = load_json(json_fname)
        z_flip = np.diag([1, 1, -1])
        for body in data["bodies"]:
            pid = body["id"]
            joints19 = rearrange(
                np.array(body["joints19"], dtype=np.float32), "(j d) -> j d", j=19, d=4
            )
            joints19[:, [0, 1, 2]] = (joints19[:, [0, 2, 1]] / 100) @ z_flip
            bodies[pid] = joints19
    except JSONDecodeError:
        bodies = {}
        logger.error(f"FAILED LOADING AT FRAME {frame} ()")
        logger.error(f"FAILED JSON FNAME: {json_fname}")
    return frame, bodies


class EntirePanopticSequence:
    """
    The generic Panoptic Sequences are coarse with
    persons walking in and out of the scene
    It is even possible that the frames are not Continous!
    """

    @staticmethod
    def is_valid_scene(scene_name: str) -> bool:
        """ """
        return isdir(
            join(
                get_data_dir(), f"panoptic_studio/{scene_name}/hdPose3d_stage1_coco19"
            )  # noqa E501
        )

    def __init__(
        self,
        scene_name: str,
        local_cache=True,
    ):
        self.skel = PANOPTIC_SKELETON
        self.scene_name = scene_name
        tmp_fname = join(
            get_cache_dir(),
            f"{scene_name}.npz",
        )
        if local_cache and isfile(tmp_fname):
            o = np.load(tmp_fname)
            self.F = o["F"]
            self.P = o["P"]
            self.M = o["M"]
        else:
            logger.info(f"EntireSequence => load from Manifold: {scene_name}")
            scene_dir = join(
                get_data_dir(), f"panoptic_studio/{scene_name}/hdPose3d_stage1_coco19"
            )
            logger.info(f"Check if we need extra 'hd' path for '{scene_dir}'")
            if isdir(join(scene_dir, "hd")):
                scene_dir = join(scene_dir, "hd")
                logger.info(f"\tExtra 'hd' path exists, moving to '{scene_dir}'...")

            logger.info(f"loading and sorting files in '{scene_dir}'")
            poses3d_files = sorted(listdir(scene_dir))
            poses3d_files = [join(scene_dir, f) for f in poses3d_files]
            logger.info(f"Found #{len(poses3d_files)} files...")
            _start = time()
            with ThreadPool(20) as p:
                all_data = p.map(load_poses3d, [fname for fname in poses3d_files])

            logger.info(
                f"EntireSequence => elapsed loading from Manifold: {time() - _start}, #{ len(all_data)}"  # noqa E501
            )
            n_frames = len(all_data)
            n_max_person = 0
            for _, bodies in all_data:
                if len(bodies) > 0:
                    n_max_person = max(n_max_person, max(bodies.keys()))

            F = np.empty((n_frames,), dtype=np.int64)
            P = np.zeros((n_frames, n_max_person + 1, 19, 4), dtype=np.float32)
            M = np.zeros((n_frames, n_max_person + 1), dtype=np.int64)

            for i, (frame, bodies) in enumerate(all_data):
                F[i] = frame
                for pid, pose3d in bodies.items():
                    P[i, pid] = pose3d.reshape((19, 4))
                    M[i, pid] = 1

            self.F = F
            self.P = P
            self.M = M
            if local_cache:
                np.savez(tmp_fname, F=F, P=P, M=M)

        self.frame2index = {}
        self.index2frame = {}
        for i, frame in enumerate(self.F):
            self.frame2index[frame] = i
            self.index2frame[i] = frame

        self.interpolate_missing_limbs()

    def interpolate_missing_limbs(self):
        """ """
        n_persons = self.P.shape[1]
        with ThreadPool(n_persons) as p:
            PM_new = p.starmap(
                interpolate_missing_limbs,
                [(self.P[:, i], self.M[:, i], i) for i in range(n_persons)],
            )
            P_new = np.array([P for P, _ in PM_new], dtype=np.float32)
            M_new = np.array([M for _, M in PM_new], dtype=np.int64)
            self.P = rearrange(P_new, "p t j d -> t p j d")
            self.M = rearrange(M_new, "p t -> t p")

    def __len__(self):
        return len(self.P)

    def split_into_scenes(
        self, *, min_length=30, force_algorithm=False
    ) -> List[PanopticScene]:  # noqa E501
        scenes = []
        for sequ in self.split_into_sequences(force_algorithm=force_algorithm):
            scenes += sequ.split_into_scenes_with_same_amount_of_persons()
        return [scene for scene in scenes if len(scene) > min_length]

    def split_into_sequences(
        self, force_algorithm=False
    ) -> List[PanopticSequence]:  # noqa E501
        """ """
        sequences = []

        # check if we have manual annotations
        global SEQ_TO_ACTUAL_FRAMES
        manual_scene_names = {}
        for name in SEQ_TO_ACTUAL_FRAMES.keys():
            scene_name = name[: name.find("|")]
            if scene_name not in manual_scene_names:
                manual_scene_names[scene_name] = []
            manual_scene_names[scene_name].append(name)

        if self.scene_name in manual_scene_names and not force_algorithm:
            for name in sorted(manual_scene_names[self.scene_name]):
                start_frame, end_frame = SEQ_TO_ACTUAL_FRAMES[name]
                start = self.frame2index[start_frame]
                end = self.frame2index[end_frame]
                pids = reduce(self.M[start : end + 1], "t n -> n", "sum").nonzero()[0]
                if len(pids) == 0:  # assert len(pids) > 0
                    raise ValueError("No person...")
                P = np.ascontiguousarray(self.P[start : end + 1, pids, :, :3])
                M = np.ascontiguousarray(self.M[start : end + 1, pids])
                sequences.append(PanopticSequence(P=P, M=M, name=name))
            return sequences

        segments = frames2segments(reduce(self.M, "t n -> t", "sum").nonzero()[0])
        for idx, (start, end) in enumerate(segments):
            pids = reduce(self.M[start : end + 1], "t n -> n", "sum").nonzero()[0]
            if len(pids) == 0:  # assert len(pids) > 0
                raise ValueError("No person...")
            P = np.ascontiguousarray(self.P[start : end + 1, pids, :, :3])
            M = np.ascontiguousarray(self.M[start : end + 1, pids])
            name = self.scene_name + f"|{idx}"
            sequences.append(PanopticSequence(P=P, M=M, name=name))
        return sequences


# === PREPROCESSING ===


def interpolate_missing_poses(seq, m, segs, pid):
    for i in range(len(segs) - 1):
        left = segs[i, 1]
        right = segs[i + 1, 0]
        if right - left == 2:
            seq[left + 1] = (seq[left] + seq[right]) / 2
            m[left + 1] = 1
        else:
            print(f"\n@pid {pid}, nopy nope", segs.shape, left, right)

    return seq, m


def interpolate_missing_limbs(seq, m, pid):
    """
    :param {n_frames x 19 x 4}
    :m {n_frames}
    """
    segs = np.array(frames2segments(m.nonzero()[0]), dtype=np.int64)

    if len(segs) > 1:
        seq, m = interpolate_missing_poses(seq=seq, m=m, segs=segs, pid=pid)
        segs = np.array(frames2segments(m.nonzero()[0]), dtype=np.int64)

    if len(segs) != 1:
        raise ValueError(
            f"afaik all persons are entering only once... {segs.shape} @pid{pid}"
        )  # noqa E501

    important_jid_connections = {
        3: np.array([0, 9], dtype=np.int64),
        4: np.array([3], dtype=np.int64),
        5: np.array([4], dtype=np.int64),
        9: np.array([0, 3], dtype=np.int64),
        10: np.array([9], dtype=np.int64),
        11: np.array([10], dtype=np.int64),
        6: np.array([2, 12], dtype=np.int64),
        7: np.array([6], dtype=np.int64),
        8: np.array([7], dtype=np.int64),
        12: np.array([2, 6], dtype=np.int64),
        13: np.array([12], dtype=np.int64),
        14: np.array([13], dtype=np.int64),
        15: np.array([0, 1, 16, 17, 18], dtype=np.int64),
        16: np.array([0, 1, 15, 17, 18], dtype=np.int64),
        17: np.array([0, 1, 15, 16, 18], dtype=np.int64),
        18: np.array([0, 1, 15, 16, 17], dtype=np.int64),
    }

    for target_jid, neighbors in important_jid_connections.items():
        seq = _mark_weird_joints_as_missing(seq, segs, target_jid, neighbors)

    seq = _interpolate_missing_limbs(seq, segs)
    seq = _head_velocity_tracking(seq, segs)
    return seq, m


@nb.njit(nb.float32[:, :, :](nb.float32[:, :, :], nb.int64[:, :]), nogil=True)
def _head_velocity_tracking(seq, segments):
    head_jids = np.array([1, 18, 17, 16, 15], dtype=np.int64)
    n_segs = len(segments)
    vel = np.zeros((3,), dtype=np.float32)
    vel_hits = np.zeros((1,), dtype=np.float32)
    for i in range(n_segs):
        start = segments[i, 0]
        end = segments[i, 1]
        for t in range(start + 1, end + 1):
            vel = vel * 0  # reset velocity
            vel_hits = vel_hits * 0
            for jid in head_jids:
                if seq[t, jid, 3] > 0.08 and seq[t - 1, jid, 3] > 0.08:
                    a = seq[t, jid, :3]
                    b = seq[t - 1, jid, :3]
                    dirvec = b - a
                    vel = vel + dirvec
                    vel_hits = vel_hits + 1
            if vel_hits[0] > 0:
                vel = vel / vel_hits
            for jid in head_jids:
                if seq[t, jid, 3] < 0.1:
                    seq[t, jid, :3] = seq[t - 1, jid, :3] - vel
    return seq


@nb.njit(nb.float32[:, :, :](nb.float32[:, :, :], nb.int64[:, :]), nogil=True)
def _interpolate_missing_limbs(seq, segments):
    """
    :param seq: {n_frames x 19 x 4}
    :param segments: [(start, end)]
    """
    n_segs = len(segments)
    for i in range(n_segs):
        start = segments[i, 0]
        end = segments[i, 1]
        for jid in range(19):
            last_good_t = -1
            for t in range(start, end + 1):
                if seq[t, jid, 3] > 0:
                    last_good_t = t
                else:
                    for t_lookahead in range(t + 1, end + 1):
                        if seq[t_lookahead, jid, 3] > 0:
                            dist = float(t_lookahead - last_good_t)
                            dist_left = float(t - last_good_t)
                            dist_right = float(t_lookahead - t)

                            w_left = dist_right / dist
                            w_right = dist_left / dist

                            for d in range(3):
                                seq[t, jid, d] = (
                                    seq[last_good_t, jid, d] * w_left
                                    + seq[t_lookahead, jid, d] * w_right
                                )
                            break
    return seq


@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.int64[:, :], nb.int64, nb.int64[:]),
    nogil=True,
)
def _mark_weird_joints_as_missing(seq, segments, target_jid, neighbor_jids):
    """
    :param seq: {n_frames x 19 x 4}
    :param segments: [(start, end)]
    """
    n_segs = len(segments)
    n_neighbors = len(neighbor_jids)

    hits = np.zeros((n_neighbors), dtype=nb.float64)
    acc_distances = np.zeros((n_neighbors), dtype=nb.float64)

    # step 0: kick out all values that are below 0.1
    for i in range(n_segs):
        start = segments[i, 0]
        end = segments[i, 1]
        for t in range(start, end + 1):
            for jid in range(19):
                if seq[t, jid, 3] < 0.08:
                    seq[t, jid, 3] = -1

    # step 1: calculate average dist
    for i in range(n_segs):
        start = segments[i, 0]
        end = segments[i, 1]
        for t in range(start, end + 1):
            if seq[t, target_jid, 3] > 0:
                pt1 = seq[t, target_jid, :3]
                for n, neighbor_jid in enumerate(neighbor_jids):
                    if seq[t, neighbor_jid, 3] > 0:
                        pt2 = seq[t, neighbor_jid, :3]
                        d = la.norm(pt2 - pt1)
                        acc_distances[n] += d
                        hits[n] += 1

        # step 2: mark as "weird" if neighbor distance too large
        leeway = 1.3
        for n in range(n_neighbors):
            if hits[n] > 0:
                acc_distances[n] = (acc_distances[n] / hits[n]) * leeway

        weird_factor_step = 1.0 / n_neighbors
        for i in range(n_segs):
            start = segments[i, 0]
            end = segments[i, 1]
            for t in range(start, end + 1):
                if seq[t, target_jid, 3] > 0:
                    pt1 = seq[t, target_jid, :3]

                    weird_factor = 0.0  # not weird <-- 0....0.5....1 --> weird
                    for n, neighbor_jid in enumerate(neighbor_jids):
                        if hits[n] > 0:
                            if seq[t, neighbor_jid, 3] > 0:
                                if seq[t, neighbor_jid, 3] > 0:
                                    pt2 = seq[t, neighbor_jid, :3]
                                    d = la.norm(pt2 - pt1)
                                    if d > acc_distances[n]:
                                        weird_factor += weird_factor_step
                                else:
                                    weird_factor += weird_factor_step * 0.5
                    if weird_factor > 0.5:
                        seq[t, target_jid, 3] = -1
    return seq


# some of the sequences are bady cut: we fix that here
SEQ_TO_ACTUAL_FRAMES = {
    "160224_haggling1|0": (224, 2129),
    "160224_haggling1|1": (2487, 4342),
    "160224_haggling1|2": (4706, 6831),
    "160224_haggling1|3": (7109, 8761),
    "160226_haggling1|0": (130, 1973),
    "160226_haggling1|1": (2224, 4035),
    "160226_haggling1|2": (4226, 5715),
    "160226_haggling1|3": (5905, 7902),
    "160226_haggling1|4": (8187, 10009),
    "160226_haggling1|5": (10206, 11593),
    "160422_haggling1|0": (333, 2501),
    "160422_haggling1|1": (2811, 4722),
    "160422_haggling1|2": (4958, 7024),
    "160422_haggling1|3": (7188, 8943),
    "160422_haggling1|4": (9149, 10936),
    "160422_haggling1|5": (11290, 13185),
    "160906_band1|0": (185, 1939),
    "160906_band3|0": (162, 7500),
    "161029_build1|0": (191, 9005),
    "161029_piano1|0": (1787, 7531),  # cut out the child
    "161029_sports1|0": (258, 730),
    "161029_sports1|1": (778, 2677),
    "161029_sports1|2": (2755, 4492),
    "161029_sports1|3": (4905, 5371),
    "161029_sports1|4": (5590, 6045),
    "161029_sports1|5": (6605, 7219),
    "161029_sports1|6": (7904, 8077),
    "161029_sports1|7": (8205, 8976),
    "161029_tools1|0": (144, 5523),
    "170221_haggling_b1|0": (1097, 3244),
    "170221_haggling_b1|1": (3518, 5409),
    "170221_haggling_b1|2": (5741, 7885),
    "170221_haggling_b1|3": (8221, 10333),
    "170221_haggling_b1|4": (10572, 12655),
    "170221_haggling_b1|5": (12929, 15055),
    "170221_haggling_b3|0": (255, 2425),
    "170221_haggling_b3|1": (2576, 4647),
    "170221_haggling_b3|2": (4915, 7023),
    "170221_haggling_b3|3": (8218, 8506),  # POSES!
    "170221_haggling_b3|4": (8909, 9521),  # POSES!
    "170221_haggling_b3|5": (9699, 10263),  # POSES!
    "170221_haggling_b3|6": (10464, 10961),  # POSES!
    "170221_haggling_b3|6": (11161, 11732),  # POSES!
    "170221_haggling_b3|7": (11936, 12466),  # POSES!
    "170221_haggling_b3|8": (12590, 13151),  # POSES!
    "170221_haggling_b3|9": (13388, 13879),  # POSES!
    "170221_haggling_b3|10": (14058, 14605),  # POSES!
    "170221_haggling_b3|11": (14681, 15115),  # POSES!
}