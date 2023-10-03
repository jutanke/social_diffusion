from os.path import join

import matplotlib.pylab as plt
import numpy as np
from einops import rearrange
from mpl_toolkits import mplot3d  # noqa E501 ensure that we can plot 3d
from social_diffusion.skeleton import Skeleton
from social_diffusion.vis import create_vis_path_local, to_mp4
from tqdm import tqdm


def plot_seq(
    seq,
    path: str,
    skel: Skeleton,
    *,
    predict_from_frame: int = -1,
    to_video: bool = True,
    figsize: int = 10,
    noaxis: bool = True,
    is_group: bool = False,
    alpha: float = 1.0,
    linewidth: float = 3.0,
    keep_past_frames: bool = False,
    plot_jids: bool = False,
    mark_origin=False,
):
    """
    :param seq: {n_frames x jd}
    """
    create_vis_path_local(path)
    if is_group:
        if len(seq.shape) == 3 or len(seq.shape) == 4:
            Seq = []
            n_person = seq.shape[1]
            for pid in range(n_person):
                Seq.append(skel.fix_shape(seq[:, pid], unroll_jd=True))
            seq = rearrange(
                np.array(Seq, dtype=np.float32), "p t j d -> t p j d"
            )  # noqa E501
        else:
            raise ValueError(f"Weird shape for group: {seq.shape}")
        seq_flat = rearrange(seq, "p t j d -> (p t) j d")
    else:
        seq = skel.fix_shape(seq, unroll_jd=True)
        seq_flat = seq
    n_frames = len(seq)

    if predict_from_frame == -1:
        predict_from_frame = n_frames + 1

    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111, projection="3d")

    if keep_past_frames:
        alphas = np.linspace(0.00001, alpha, num=n_frames)
    else:
        alphas = [alpha] * n_frames

    for t in tqdm(range(n_frames), total=n_frames, leave=True, position=0):
        if not keep_past_frames:
            ax.clear()
            ax.set_title(f"frame {t}")
        if noaxis:
            ax.axis("off")

        if mark_origin:
            ax.plot([-1, 1], [0, 0], [0, 0], color="black", alpha=0.5)
            ax.plot([0, 0], [-1, 1], [0, 0], color="black", alpha=0.5)
            ax.plot([0, 0], [0, 0], [-1, 1], color="black", alpha=0.5)

        set_axis(ax, seq_flat, skel)
        if t >= predict_from_frame:
            lcolor = "orange"
            rcolor = "green"
        else:
            lcolor = "cornflowerblue"
            rcolor = "salmon"
        plot_pose(
            ax,
            seq[t],
            skel,
            left_color=lcolor,
            right_color=rcolor,
            alpha=alphas[t],
            linewidth=linewidth,
            plot_jids=plot_jids,
        )
        fname = join(path, "frame%05d.png" % t)
        plt.tight_layout()
        plt.savefig(fname)

    if to_video:
        to_mp4(path)


def set_axis(ax, pose, skel: Skeleton):
    """
    :param ax:
    :param pose: {n_person x jd}
    :param skel: Skel
    """
    pose = skel.fix_shape(pose, unroll_jd=True)  # p x j x d
    X = pose[:, :, 0]
    Y = pose[:, :, 1]
    Z = pose[:, :, 2]

    Xmin = np.min(X)
    Ymin = np.min(Y)
    Zmin = np.min(Z)
    Xmax = np.max(X)
    Ymax = np.max(Y)
    Zmax = np.max(Z)

    max_size = max(Xmax - Xmin, max(Ymax - Ymin, Zmax - Zmin))
    ax.set_xlim(
        [(Xmax + Xmin) / 2 - max_size / 2, (Xmax + Xmin) / 2 + max_size / 2]
    )  # noqa E501
    ax.set_ylim(
        [(Ymax + Ymin) / 2 - max_size / 2, (Ymax + Ymin) / 2 + max_size / 2]
    )  # noqa E501
    ax.set_zlim(
        [(Zmax + Zmin) / 2 - max_size / 2, (Zmax + Zmin) / 2 + max_size / 2]
    )  # noqa E501


def plot_pose(
    ax,
    pose,
    skel: Skeleton,
    *,
    left_color="cornflowerblue",
    right_color="salmon",
    center_color="gray",
    plot_jids=False,
    alpha=1.0,
    linewidth=3,
):
    """
    :param ax:
    :param pose: {n_person x jd}
    :param skel: Skel
    """
    pose = skel.fix_shape(pose, unroll_jd=True)  # p x j x d
    Alpha = None
    if isinstance(alpha, list) or isinstance(alpha, np.ndarray):
        Alpha = alpha
        if len(Alpha) != len(pose):
            raise ValueError(
                f"Alpha ({len(Alpha)}) and pose ({len(pose)}) have to have the same length"  # noqa E501
            )

    n = len(pose)
    if Alpha is None:
        Alpha = [alpha for _ in range(n)]
    for a, b in skel.skeleton:
        if a in skel.left_jids and b in skel.left_jids:
            color = left_color
        elif a in skel.right_jids and b in skel.right_jids:
            color = right_color
        else:
            color = center_color

        for t in range(n):
            px = pose[t, a, 0]
            py = pose[t, a, 1]
            pz = pose[t, a, 2]

            qx = pose[t, b, 0]
            qy = pose[t, b, 1]
            qz = pose[t, b, 2]

            ax.plot(
                [px, qx],
                [py, qy],
                [pz, qz],
                color=color,
                linewidth=linewidth,
                alpha=Alpha[t],
            )

    if plot_jids:
        for t in range(n):
            for jid, loc in enumerate(pose[t]):
                ax.text(loc[0], loc[1], loc[2], str(jid))
