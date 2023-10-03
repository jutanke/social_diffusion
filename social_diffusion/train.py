import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from social_diffusion import (
    create_training_dir,
    get_device,
    get_number_of_processes,
)  # noqa E501
from social_diffusion.diffusion import GaussianDiffusion

from social_diffusion.model import MotionModel
from social_diffusion.ema import EMA
from os.path import join, isdir

# from os import listdir
import numpy as np


def load_socdiff(
    model_dir: str,
    n_in: int,
    n_frames: int,
    hidden_dim: int,
    epoch: int,
    skel,
    n_person: int,
    device,
    *,
    objective="pred_x0",
):
    assert isdir(model_dir)
    fname_model = join(model_dir, f"model_{epoch}.pt")
    fname_ema = join(model_dir, f"ema_{epoch}.pt")

    model = MotionModel(
        n_in=n_in, n_frames=n_frames, skel=skel, hidden_dim=hidden_dim
    )  # noqa E501
    diffusion = GaussianDiffusion(
        model,
        n_person=n_person,
        n_in=n_in,
        n_frames=n_frames,
        objective=objective,
    ).to(device)
    ema_update_every = 10
    ema_decay = 0.995
    ema_diffusion = EMA(
        diffusion, beta=ema_decay, update_every=ema_update_every
    )  # noqa E501

    model.load_state_dict(torch.load(fname_model))
    ema_diffusion.load_state_dict(torch.load(fname_ema))

    return model, ema_diffusion


def train_socdiff(
    model_dir: str,
    ds_train,
    n_in,
    n_frames,
    n_person,
    *,
    batch_size=64,
    epochs=50,
    hidden_dim=256,
    device=None,
    tqdm_over_epochs=False,
    save_optimizer=False,
):
    """
    Train SocialDiffusion
    :param transform_batch_fn: def transform_batch_fn(batch) -> batch
    :param extra_loss_fn: def extra_loss_fn(batch, model_out, ds_train) ->
        loss (scalar)
    """
    train_lr = 1e-4

    print(f"Train SocDiff: {model_dir}")
    create_training_dir(model_dir)

    print(f"#train: {len(ds_train):,}")

    if device is None:
        device = get_device()
    objective = "pred_x0"

    model = MotionModel(
        n_in=n_in, n_frames=n_frames, skel=ds_train.skel, hidden_dim=hidden_dim
    )
    num_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    # logger.info(f"#params: {num_params:,} (hidden: {hidden_dim})")
    print(f"#params: {num_params:,} (hidden: {hidden_dim})")

    dl = DataLoader(
        ds_train,
        batch_size=batch_size,
        num_workers=get_number_of_processes(),
        shuffle=True,
    )

    adam_betas = (0.9, 0.99)
    diffusion = GaussianDiffusion(
        model,
        n_person=n_person,
        n_in=n_in,
        n_frames=n_frames,
        objective=objective,
    ).to(device)

    opt = Adam(diffusion.parameters(), lr=train_lr, betas=adam_betas)

    ema_update_every = 10
    ema_decay = 0.995
    ema_diffusion = EMA(
        diffusion, beta=ema_decay, update_every=ema_update_every
    )  # noqa E501

    for epoch in (
        pbar_epoch := tqdm(
            range(0, epochs),
            position=0,
            leave=True,
            disable=not tqdm_over_epochs,  # noqa E501
        )
    ):
        total_losses = []
        for batch in (
            pbar := tqdm(dl, position=0, leave=True, disable=tqdm_over_epochs)
        ):
            for k in batch.keys():
                batch[k] = batch[k].to(device)

            loss, model_out = diffusion(batch)

            loss.backward()

            # stupid pytorch bug...
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            opt.step()
            opt.zero_grad()

            total_losses.append(loss.item())

            if tqdm_over_epochs:
                pbar_epoch.set_description(
                    "epoch %04d loss %0.5f" % (epoch, np.mean(total_losses))
                )
            else:
                pbar.set_description(
                    "epoch %04d loss %0.5f" % (epoch, np.mean(total_losses))
                )

            ema_diffusion.update()

        fname_model = join(model_dir, f"model_{epoch}.pt")
        fname_ema = join(model_dir, f"ema_{epoch}.pt")
        fname_opt = join(model_dir, f"opt_{epoch}.pt")
        torch.save(model.state_dict(), fname_model)
        torch.save(ema_diffusion.state_dict(), fname_ema)
        if save_optimizer:
            torch.save(opt.state_dict(), fname_opt)

    ema_diffusion.eval()
    diffusion.eval()
    return {
        "model": model,
        "diffusion": diffusion,
        "ema_diffusion": ema_diffusion,
        "ds_train": ds_train,
        "n_in": n_in,
        "n_frames": n_frames,
    }