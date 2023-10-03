from os.path import join
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import reduce, rearrange
from social_diffusion.transforms.transforms import (
    get_normalization,
    apply_normalization_to_seq,
)


class WhoIsBuyerDecider:
    def __init__(self, n_frames=512, device="cuda"):
        """ """
        self.n_frames = n_frames
        fname_local = join(os.getcwd(), "models/whoisbuyer.ckpt")

        self.model = WhoIsTheBuyerModule.load_from_checkpoint(fname_local)
        self.model.eval().to(device)
        self.device = device

    def who_is_the_buyer(self, P, ss=50):
        """
        :param P: {n_frames x 3 x 57}
        """
        selected = []
        n_frames = len(P)
        for t in range(0, n_frames - self.n_frames, ss):
            P_chunk = P[t : t + self.n_frames]  # noqa E203
            pid = self.eval_chunk(P_chunk)
            selected.append(pid)

        U, C = np.unique(selected, return_counts=True)
        best_u = -1
        best_c = -1
        for u, c in zip(U, C):
            if c > best_c:
                best_c = c
                best_u = u
        return best_u

    def eval_chunk(self, P):
        """
        :param P: {n_frames x 3 x 57}
        """
        assert (
            len(P.shape) == 3
            and P.shape[0] == self.n_frames
            and P.shape[1] == 3
            and P.shape[2] == 57
        ), f"weird P shape: {P.shape}"

        index_rearrange, B = ensure_clockwise_order(P)

        B = torch.from_numpy(B).to(self.device)

        with torch.no_grad():
            pred = self.model.model(B).cpu().numpy()

        return index_rearrange[np.argmax(pred)]


def ensure_clockwise_order(P):
    """
    :param P: {n_frames x 3 x 57}
    :returns:
        {3 x n_frames x 3 x 57}
    """
    assert (
        len(P.shape) == 3 and P.shape[1] == 3 and P.shape[2] == 57
    ), f"weird P shape: {P.shape}"
    P = rearrange(P, "t p (j d) -> t p j d", j=19, d=3)
    A = P[:, 0]
    B = P[:, 1]
    C = P[:, 2]

    # we don't know the order (who's left/right of whom)
    # if we follow clock-wise we want the following order:
    # A --> B --> C --> A
    # step 1: figure out order and make sure its the above order!
    mean_hip = reduce(A[:, [6, 12]], "t j d -> j d", "mean")
    mu, R = get_normalization(mean_hip[0], mean_hip[1])

    B_in_A = apply_normalization_to_seq(B, mu=mu, R=R)
    C_in_A = apply_normalization_to_seq(C, mu=mu, R=R)

    # # the person with the lower x-value is the left-hand person
    mean_B = reduce(B_in_A[:, [6, 12]], "t j d -> d", "mean")
    mean_C = reduce(C_in_A[:, [6, 12]], "t j d -> d", "mean")

    index_rearrange = [0, 1, 2]

    if mean_B[0] > mean_C[0]:
        B, C = C, B
        index_rearrange[1] = 2
        index_rearrange[2] = 1

    SeqABC = normalize_data(target_seq=A, left_seq=B, right_seq=C)
    SeqBCA = normalize_data(target_seq=B, left_seq=C, right_seq=A)
    SeqCAB = normalize_data(target_seq=C, left_seq=A, right_seq=B)

    SeqABC = rearrange(SeqABC, "t d jd -> 1 t d jd")
    SeqBCA = rearrange(SeqBCA, "t d jd -> 1 t d jd")
    SeqCAB = rearrange(SeqCAB, "t d jd -> 1 t d jd")

    B = np.concatenate([SeqABC, SeqBCA, SeqCAB], axis=0)

    return index_rearrange, B


def normalize_data(target_seq, left_seq, right_seq):
    """
    :param _seq: {n_frames x 57}
    """
    target_seq = target_seq.copy()
    left_seq = left_seq.copy()
    right_seq = right_seq.copy()
    if len(target_seq.shape) == 2:
        target_seq = rearrange(target_seq, "t (j d) -> t j d", j=19)
    if len(left_seq.shape) == 2:
        left_seq = rearrange(left_seq, "t (j d) -> t j d", j=19)
    if len(right_seq.shape) == 2:
        right_seq = rearrange(right_seq, "t (j d) -> t j d", j=19)
    target_seq_norm = reduce(target_seq[:, [6, 12]], "t j d -> j d", "mean")
    hip_left = target_seq_norm[0]
    hip_right = target_seq_norm[1]
    mu, R = get_normalization(left3d=hip_left, right3d=hip_right)
    target_seq = apply_normalization_to_seq(seq=target_seq, mu=mu, R=R)
    left_seq = apply_normalization_to_seq(seq=left_seq, mu=mu, R=R)
    right_seq = apply_normalization_to_seq(seq=right_seq, mu=mu, R=R)
    target_seq = rearrange(target_seq, "t j d -> t 1 (j d)")
    left_seq = rearrange(left_seq, "t j d -> t 1 (j d)")
    right_seq = rearrange(right_seq, "t j d -> t 1 (j d)")
    return np.concatenate([target_seq, left_seq, right_seq], axis=1)


class Residual(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        if dim_in == dim_out:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv1d(dim_in, dim_out, kernel_size=1)
        self.nn = nn.Sequential(
            nn.Conv1d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                padding=1,  # noqa E501
            ),
            nn.SiLU(),
            nn.BatchNorm1d(dim_out),
            nn.Conv1d(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                padding=1,  # noqa E501
            ),
        )

    def forward(self, x):
        """
        :param x: {n_batch x 57 x n_frames}
        """
        return self.nn(x) + self.proj(x)


class Down(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.nn = nn.Sequential(
            Residual(dim_in, dim_out),
            nn.SiLU(),
            nn.BatchNorm1d(dim_out),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1, stride=2),
        )

    def forward(self, x):
        return self.nn(x)


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, max_len: int = 2000, dropout: float = 0.1
    ):  # noqa E501
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
            output: [sequence length, batch size, embed dim]
        """
        x = rearrange(x, "b d t -> t b d")
        x = x + self.pe[: x.size(0)]
        return rearrange(self.dropout(x), "t b d -> b t d")


class Model(nn.Module):
    def __init__(self, pose_dim, hidden_dim):
        super().__init__()
        self.down1_target = Down(pose_dim, hidden_dim)
        self.down1_left = Down(pose_dim, hidden_dim)
        self.down1_right = Down(pose_dim, hidden_dim)
        self.down2 = Down(hidden_dim * 3, hidden_dim * 4)
        self.down3 = Down(hidden_dim * 4, hidden_dim * 4)
        self.decision = nn.Sequential(
            Down(hidden_dim * 4, hidden_dim * 4),
            Down(hidden_dim * 4, hidden_dim * 4),
            Down(hidden_dim * 4, hidden_dim * 8),
            Down(hidden_dim * 8, hidden_dim * 8),
            nn.Conv1d(
                in_channels=hidden_dim * 8,
                out_channels=1,
                kernel_size=4,
                padding=0,  # noqa E501
            ),
        )

    def forward(self, Seq):
        """
        :param Seq: {n_batch x n_frames x 3 x 57}
        """
        target_seq = rearrange(Seq[:, :, 0], "b t d -> b d t")
        left_seq = rearrange(Seq[:, :, 1], "b t d -> b d t")
        right_seq = rearrange(Seq[:, :, 2], "b t d -> b d t")

        down1_target = self.down1_target(target_seq)
        down1_left = self.down1_left(left_seq)
        down1_right = self.down1_right(right_seq)
        down1 = torch.cat([down1_target, down1_left, down1_right], dim=1)

        down2 = self.down2(down1)
        down3 = self.down3(down2)

        dd = self.decision(down3)[:, :, 0]

        return dd


class WhoIsTheBuyerModule(pl.LightningModule):
    def __init__(self, pose_dim=57, hidden_dim=64):
        super().__init__()
        self.model = Model(pose_dim, hidden_dim)
        self.logit_bce_loss = nn.BCEWithLogitsLoss()

    def step(self, Seq, is_buyer):
        pred_logits = self.model(Seq)
        loss = self.logit_bce_loss(pred_logits, is_buyer)
        return pred_logits, loss

    def training_step(self, batch, batch_idx):
        Seq, is_buyer = batch
        is_buyer = is_buyer.unsqueeze(1)
        is_buyer_f = is_buyer.float()
        pred_logits, loss = self.step(Seq, is_buyer_f)
        with torch.no_grad():
            accuracy = torch.mean(
                (
                    (((F.sigmoid(pred_logits) > 0.5) * 1) == is_buyer) * 1
                ).float()  # noqa E501
            )

        self.log("acc", accuracy, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss, "acc": accuracy}

    def validation_step(self, batch, batch_idx):
        Seq, is_buyer = batch
        is_buyer = is_buyer.unsqueeze(1)
        is_buyer_f = is_buyer.float()
        pred_logits, loss = self.step(Seq, is_buyer_f)

        with torch.no_grad():
            accuracy = torch.mean(
                (
                    (((F.sigmoid(pred_logits) > 0.5) * 1) == is_buyer) * 1
                ).float()  # noqa E501
            )

        self.log(
            "val_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True
        )  # noqa E501
        self.log(
            "val_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True
        )  # noqa E501
        return {"val_loss": loss.item(), "val_acc": accuracy}