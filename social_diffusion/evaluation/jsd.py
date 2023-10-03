import math as m
import numpy as np
import numpy.linalg as la
import torch
from einops import rearrange, reduce
from scipy.ndimage import gaussian_filter1d
from social_diffusion.transforms.transforms import (
    get_normalization,
    apply_normalization_to_seq,
)


def get_hm(words, size=16):
    HM = np.zeros((size, size), dtype=np.float32)
    for t in range(len(words) - 1):
        a = words[t]
        b = words[t + 1]
        HM[a, b] += 1
    return HM


def to_prob(hm):
    hm = rearrange(hm, "w h -> (w h)")
    return hm / np.sum(hm)


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


def calculate_head_orientations(poses):
    """
    :param poses: {n_frames x 3 x 57}
    :return
            {n_frames x n_person x 2 x 2}  # (pos / norm)
    """
    n_person = 3
    n_frames = poses.shape[0]
    head_orientations = np.empty((n_frames, n_person, 2, 2), dtype=np.float32)
    for i in range(n_person):
        seq = poses[:, i]
        head_orientations[:, i, 0] = rearrange(
            seq, "t (j d) -> t j d", j=19, d=3
        )[  # noqa E501
            :, 1, :2
        ]
        head_orientations[:, i, 1] = get_normal(seq, left_jid=15, right_jid=17)
    return head_orientations


def calculate_attn(P, buyer_id):
    """
    :param P: {n_frames x 3 x 57}
    """
    n_frames = P.shape[0]
    HO = calculate_head_orientations(P)
    seller_ids = [pid for pid in range(3) if pid != buyer_id]
    seller1 = HO[:, seller_ids[0]]
    seller2 = HO[:, seller_ids[1]]
    buyer = HO[:, buyer_id]

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

    attn = np.zeros((n_frames, 2), dtype=np.int64)
    for t in range(n_frames):
        if att1[t] > att2[t]:
            attn[t, 0] = 1
        else:
            attn[t, 1] = 1
    return attn, seller_ids


def get_left_right_seller_ids(P, buyer_id):
    """
    :param P: {n_frames x 3 x 57}
    :return left id, right id
    """
    seller_ids = [pid for pid in range(3) if pid != buyer_id]
    buyer = P[:, buyer_id]
    seller1 = rearrange(P[:, seller_ids[0]], "t (j d) -> t j d", j=19, d=3)
    seller2 = rearrange(P[:, seller_ids[1]], "t (j d) -> t j d", j=19, d=3)

    buyer_mean_hip = reduce(
        buyer, "t (j d) -> j d", j=19, d=3, reduction="mean"
    )  # noqa E501

    mu, R = get_normalization(buyer_mean_hip[6], buyer_mean_hip[12])

    seller1_trans = apply_normalization_to_seq(seller1, mu=mu, R=R)
    seller2_trans = apply_normalization_to_seq(seller2, mu=mu, R=R)

    mean_seller1 = reduce(
        seller1_trans, "t j d -> d", j=19, d=3, reduction="mean"
    )  # noqa E501
    mean_seller2 = reduce(
        seller2_trans, "t j d -> d", j=19, d=3, reduction="mean"
    )  # noqa E501

    if mean_seller1[0] > mean_seller2[0]:
        return seller_ids[1], seller_ids[0]
    else:
        return seller_ids[0], seller_ids[1]


def attn_to_word5(attn, speech, buyer_speech):
    """
    :param attn: {n_frames x 2}
    :param speech: {n_frames x 2}
    :param buyer_speech: {n_frames}
    :param left_pid/right_pid  --> scaled to fit attn!!!
    """
    # 0 --> [0, l]    8 --> [b, l]
    # 1 --> [0, r]    9 --> [b, r]
    # 2 --> [L, l]   10 --> [bL, l]
    # 3 --> [L, r]   11 --> [bL, r]
    # 4 --> [R, l]   12 --> [bR, l]
    # 5 --> [R, r]   13 --> [bR, r]
    # 6 --> [LR, l]  14 --> [bLR, l]
    # 7 --> [LR, r]  15 --> [bLR, r]
    word_lookup_list = [
        ",l",
        ",r",
        "L,l",
        "L,r",
        "R,l",
        "R,r",
        "LR,l",
        "LR,r",
        "b,l",
        "b,r",
        "Lb,l",
        "Lb,r",
        "Rb,l",
        "Rb,r",
        "LRb,l",
        "LRb,r",
    ]
    assert len(word_lookup_list) == 16
    word_lookup = {}
    for i, word in enumerate(word_lookup_list):
        assert word not in word_lookup
        word_lookup[word] = i

    words = []
    for t in range(len(attn)):
        word = ""
        # -- step 1 -- handle who speaks
        if speech[t, 0] > 0.5:
            word += "L"
        if speech[t, 1] > 0.5:
            word += "R"
        if buyer_speech[t] > 0.5:
            word += "b"
        word += ","
        # -- step 2 -- handle attn
        if attn[t, 0] > 0.5:
            word += "l"
        elif attn[t, 1] > 0.5:
            word += "r"
        else:
            raise ValueError(f"wtf happend to the attention? @frame:{t}")
        words.append(word_lookup[word])
    return words


def words2segments(words, scene_name="None"):
    segments = []
    current_label = words[0]
    count = 1
    for i in range(1, len(words)):
        if words[i] == current_label:
            count += 1
        else:
            segments.append(
                {
                    "label": current_label,
                    "count": count,
                    "scene_name": scene_name,
                    "t": i - count,
                }
            )
            current_label = words[i]
            count = 1
    segments.append(
        {
            "label": current_label,
            "count": count,
            "scene_name": scene_name,
            "t": i - count,
        }
    )  # add final segment
    return segments