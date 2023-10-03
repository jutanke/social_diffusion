import json
from typing import List
import numpy as np
from einops import rearrange, reduce


def calculate_standardization(all_poses, skel):
    """
    :param all_poses: {b t p jd}
    """
    eps = 0.000000001
    all_poses = rearrange(all_poses, "b t p jd -> (b p t) jd")
    p_mu = reduce(all_poses, "t jd -> jd", "mean")
    p_std = np.std(all_poses, axis=0) + eps
    return p_mu, p_std


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def frames2segments(frames: List[int], return_indices=False, include_length=False):
    """
    :param frames: [1, 2, 3, 10, 11, 12, 13, ...]
    :return
        (1, 3), (11, 13), ..
    """
    segments = []
    indices = []
    if len(frames) > 0:
        frame = frames[0]
        i_start = 0
        for i in range(1, len(frames)):
            if frames[i - 1] + 1 < frames[i]:
                if include_length:
                    segments.append((frame, frames[i - 1], 1 + frames[i - 1] - frame))
                else:
                    segments.append((frame, frames[i - 1]))
                indices.append((i_start, i - 1))
                frame = frames[i]
                i_start = i
        # handle the last segment
        if include_length:
            segments.append((frame, frames[-1], 1 + frames[-1] - frame))
        else:
            segments.append((frame, frames[-1]))
        indices.append((i_start, len(frames) - 1))

    if return_indices:
        assert len(indices) == len(segments)
        return segments, indices

    return segments