from social_diffusion import get_output_dir
from social_diffusion.vis.plot import plot_seq
from social_diffusion.train import load_socdiff
from social_diffusion.datasets.panoptic import PANOPTIC_SKELETON
from os.path import join
import numpy as np
from multiprocessing import Pool
from social_diffusion.datasets.haggling import (
    get_haggling_local_dataset,
    get_haggling_test_sequences,
)
import torch
from einops import repeat


def render_seq(entry):
    Seq = entry["Seq"]
    vis_path = entry["vis_path"]
    plot_seq(
        seq=Seq,
        path=vis_path,
        skel=PANOPTIC_SKELETON,
        is_group=True,
        predict_from_frame=178,
    )


def vis():
    n_in = 128
    n_frames = 512
    n_person = 3
    hidden_dim = 128 + 64

    model_dir = join(
        get_output_dir(),
        f"models/socdiff_haggling_nin{n_in}_nframes{n_frames}_hidden{hidden_dim}",  # noqa E501
    )
    device = torch.device("cuda")

    ds_train, _, skel = get_haggling_local_dataset(
        n_frames=n_frames, n_in=n_in
    )  # noqa E501

    _, ema_diffusion = load_socdiff(
        model_dir=model_dir,
        n_in=n_in,
        n_frames=n_frames,
        hidden_dim=hidden_dim,
        epoch=50,
        skel=skel,
        n_person=n_person,
        device=device,
    )

    test_seqs = get_haggling_test_sequences()

    seq_idx = 0
    n_samples = 8
    JUMPOFF_FRAME = 178

    test_seq = test_seqs[seq_idx]
    Seq = test_seq.Seq
    P = np.zeros((2048, 3, 57), dtype=np.float32)
    P[:128] = Seq[JUMPOFF_FRAME - 128 : JUMPOFF_FRAME]  # noqa E203
    Seq_hat = ema_diffusion.ema_model.predict(
        P, ds_train=ds_train, n_samples=n_samples
    )  # noqa E501

    Seq_hat[:, :128] = Seq[JUMPOFF_FRAME - 128 : JUMPOFF_FRAME]  # noqa E203

    Seq_hat_init = repeat(
        Seq[: JUMPOFF_FRAME - 128], "t p jd -> s t p jd", s=n_samples
    )  # noqa E501

    Seq_hat = np.concatenate([Seq_hat_init, Seq_hat], axis=1)

    Data = []
    for i in range(8):
        Seq = Seq_hat[i, :512]
        vis_path = join(get_output_dir(), f"haggling{seq_idx}/vis{i}")
        Data.append({"Seq": Seq, "vis_path": vis_path})
    with Pool(8) as p:
        p.map(render_seq, Data)


if __name__ == "__main__":
    vis()
