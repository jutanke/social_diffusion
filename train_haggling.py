from social_diffusion import get_output_dir
from os.path import join, isdir
from os import makedirs
from social_diffusion.datasets.haggling import get_haggling_test_sequences

# from social_diffusion.vis.plot import plot_seq
from social_diffusion.datasets.haggling import get_haggling_local_dataset
from social_diffusion.train import train_socdiff
import numpy as np


def train():
    print("start training on Haggling")

    n_in = 128
    n_frames = 512
    n_person = 3
    hidden_dim = 128 + 64

    # test_seqs = get_haggling_test_sequences()

    model_dir = join(
        get_output_dir(),
        f"models/socdiff_haggling_nin{n_in}_nframes{n_frames}_hidden{hidden_dim}",  # noqa E501
    )

    ds_train, _, skel = get_haggling_local_dataset(
        n_frames=n_frames, n_in=n_in
    )  # noqa E501

    training_data = train_socdiff(
        model_dir=model_dir,
        ds_train=ds_train,
        n_in=n_in,
        n_frames=n_frames,
        n_person=n_person,
        hidden_dim=hidden_dim,
    )

    test_seqs = get_haggling_test_sequences()

    OUT = []
    for test_seq in test_seqs:
        Seq = test_seq.Seq
        P = np.zeros((2048, 3, 57), dtype=np.float32)
        P[:128] = Seq[:128]
        ema_diffusion = training_data["ema_diffusion"]
        Seq_hat = ema_diffusion.ema_model.predict(
            P, ds_train=ds_train, n_samples=8
        )  # noqa E501
        OUT.append(Seq_hat)

    outpath = join(get_output_dir(), "results/haggling")
    if not isdir(outpath):
        makedirs(outpath)
    fname = join(outpath, "pred.npy")
    OUT = np.array(OUT, dtype=np.float32)
    np.save(fname, OUT)


if __name__ == "__main__":
    train()
