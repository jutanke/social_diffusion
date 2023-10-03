from os.path import join
from social_diffusion.datasets.haggling import get_haggling_local_dataset
from social_diffusion.train import load_socdiff
import numpy as np
from einops import repeat
from social_diffusion import get_output_dir
from social_diffusion.sequence import Sequence
import torch


def predict(ema_diffusion, ds_train, Seq: Sequence, *, n_samples=8):
    INPUT_FRAME_10PERC = 178

    P = np.zeros((2048, 3, 57), dtype=np.float32)
    P[:128] = Seq[INPUT_FRAME_10PERC - 128 : INPUT_FRAME_10PERC]  # noqa E203
    Seq_hat = ema_diffusion.ema_model.predict(
        P, ds_train=ds_train, n_samples=n_samples
    )  # noqa E501
    Seq_hat_init = repeat(
        Seq[: INPUT_FRAME_10PERC - 128], "t p jd -> s t p jd", s=n_samples
    )  # noqa E501
    return np.concatenate([Seq_hat_init, Seq_hat], axis=1)


def load_trained_model():
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

    return ds_train, ema_diffusion
