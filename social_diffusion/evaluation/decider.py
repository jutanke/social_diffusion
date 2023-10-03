import os
from os.path import join, isfile


import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from social_diffusion.transforms.transforms import to_canconical_form

from torch import nn, optim
from einops import rearrange

# from unio import isfile as isfile_unio, open as open_unio, UnioConfig


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


class Model(nn.Module):
    def __init__(self, pose_dim, hidden_dim, dropout=0.1, num_layers=3):
        """ """
        super().__init__()
        self.pose_encoder = nn.Linear(pose_dim, hidden_dim)
        self.gru_single = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )
        self.merge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.merge2 = nn.Linear(4 * hidden_dim, hidden_dim)

        self.gru_with_hist = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )

        self.gru_with_hist2 = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
        )

        self.intermediate = nn.Linear(2 * hidden_dim, 1)
        self.intermediate2 = nn.Linear(4 * hidden_dim, 1)
        self.final = nn.Linear(4 * hidden_dim, 1)

    def forward(self, Seq):
        """
        :param Seq: {n_batch x n_frames x 3 x 57}
        """
        emb = self.pose_encoder(Seq)

        seq0 = emb[:, :, 0]
        seq1 = emb[:, :, 1]
        seq2 = emb[:, :, 2]

        o0, _ = self.gru_single(seq0)
        o1, _ = self.gru_single(seq1)
        o2, _ = self.gru_single(seq2)

        out_interm0 = self.intermediate(o0)[:, :, 0]
        out_interm1 = self.intermediate(o1)[:, :, 0]
        out_interm2 = self.intermediate(o2)[:, :, 0]

        temp0 = self.merge(o0)
        temp1 = self.merge(o1)
        temp2 = self.merge(o2)

        combined_history = (
            temp0 + temp1 + temp2
        )  # order-invariant!  # TODO: RELU?  # noqa E501

        # TODO replace seq0 with temp0
        o0, _ = self.gru_with_hist(
            torch.cat([seq0, temp0, combined_history], dim=2)
        )  # noqa E501
        o1, _ = self.gru_with_hist(
            torch.cat([seq1, temp1, combined_history], dim=2)
        )  # noqa E501
        o2, _ = self.gru_with_hist(
            torch.cat([seq2, temp2, combined_history], dim=2)
        )  # noqa E501

        out_interm2_0 = self.intermediate2(o0)[:, :, 0]
        out_interm2_1 = self.intermediate2(o1)[:, :, 0]
        out_interm2_2 = self.intermediate2(o2)[:, :, 0]

        temp2_0 = self.merge2(o0)
        temp2_1 = self.merge2(o1)
        temp2_2 = self.merge2(o2)
        combined_history2 = temp2_0 + temp2_1 + temp2_2

        o0, _ = self.gru_with_hist2(
            torch.cat([seq0, temp2_0, combined_history2], dim=2)
        )
        o1, _ = self.gru_with_hist2(
            torch.cat([seq1, temp2_1, combined_history2], dim=2)
        )
        o2, _ = self.gru_with_hist2(
            torch.cat([seq2, temp2_2, combined_history2], dim=2)
        )

        out_final0 = self.final(o0)[:, :, 0]
        out_final1 = self.final(o1)[:, :, 0]
        out_final2 = self.final(o2)[:, :, 0]

        return (
            out_final0,
            out_final1,
            out_final2,
            out_interm0,
            out_interm1,
            out_interm2,
            out_interm2_0,
            out_interm2_1,
            out_interm2_2,
        )


class WhoIsTheBuyerModule(pl.LightningModule):
    def __init__(self, pose_dim=57, hidden_dim=128):
        super().__init__()
        self.model = Model(pose_dim, hidden_dim)
        self.logit_bce_loss = nn.BCEWithLogitsLoss()

    def step(self, Seq, speech):
        """
        :param Seq: {n_batch x n_frames x 3 x 57}
        :param speech: {n_batch x n_frames x 3}
        """
        speech0 = speech[:, :, 0]
        speech1 = speech[:, :, 1]
        speech2 = speech[:, :, 2]

        speech0_f = speech0.float()
        speech1_f = speech1.float()
        speech2_f = speech2.float()

        (
            out_final0,
            out_final1,
            out_final2,
            out_interm0,
            out_interm1,
            out_interm2,
            out_interm2_0,
            out_interm2_1,
            out_interm2_2,
        ) = self.model(Seq)

        loss0 = (
            self.logit_bce_loss(out_final0, speech0_f)
            + self.logit_bce_loss(out_interm0, speech0_f)
            + +self.logit_bce_loss(out_interm2_0, speech0_f)
        )
        loss1 = (
            self.logit_bce_loss(out_final1, speech1_f)
            + self.logit_bce_loss(out_interm1, speech1_f)
            + self.logit_bce_loss(out_interm2_1, speech1_f)
        )
        loss2 = (
            self.logit_bce_loss(out_final2, speech2_f)
            + self.logit_bce_loss(out_interm2, speech2_f)
            + self.logit_bce_loss(out_interm2_2, speech2_f)
        )

        with torch.no_grad():
            acc_interm0 = torch.mean(
                ((((F.sigmoid(out_interm0) > 0.5) * 1) == speech0) * 1).float()
            )
            acc_final0 = torch.mean(
                ((((F.sigmoid(out_final0) > 0.5) * 1) == speech0) * 1).float()
            )

            acc_interm1 = torch.mean(
                ((((F.sigmoid(out_interm1) > 0.5) * 1) == speech1) * 1).float()
            )
            acc_final1 = torch.mean(
                ((((F.sigmoid(out_final1) > 0.5) * 1) == speech1) * 1).float()
            )

            acc_interm2 = torch.mean(
                ((((F.sigmoid(out_interm2) > 0.5) * 1) == speech2) * 1).float()
            )
            acc_final2 = torch.mean(
                ((((F.sigmoid(out_final2) > 0.5) * 1) == speech2) * 1).float()
            )

            acc_final = (acc_final0 + acc_final1 + acc_final2) / 3.0
            acc_interm = (acc_interm0 + acc_interm1 + acc_interm2) / 3.0

        loss = loss0 + loss1 + loss2

        return loss, acc_final, acc_interm

    def training_step(self, batch, batch_idx):
        Seq, speech = batch
        loss, acc_final, acc_interm = self.step(Seq, speech)
        self.log(
            "acc_final", acc_final, on_step=True, on_epoch=True, prog_bar=True
        )  # noqa E501
        self.log(
            "acc_interm",
            acc_interm,
            on_step=True,
            on_epoch=True,
            prog_bar=True,  # noqa E501
        )  # noqa E501
        return {"loss": loss, "acc_final": acc_final, "acc_interm": acc_interm}

    def validation_step(self, batch, batch_idx):
        Seq, speech = batch
        loss, acc_final, acc_interm = self.step(Seq, speech)
        self.log(
            "val_acc_final",
            acc_final,
            on_step=False,
            on_epoch=True,
            prog_bar=True,  # noqa E501
        )
        self.log(
            "val_acc_interm",
            acc_interm,
            on_step=False,
            on_epoch=True,
            prog_bar=True,  # noqa E501
        )
        return {
            "val_loss": loss,
            "val_acc_final": acc_final,
            "val_acc_interm": acc_interm,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer

    def eval_chunk(self, P):
        """
        :param P: {n_frames x 3 x 57}
        """

        # P_canon = np.concatenate([seq0, seq1, seq2], axis=1)
        # P_canon = torch.from_numpy(P_canon).unsqueeze(0).to(self.device)
        with torch.no_grad():
            P = torch.from_numpy(P).to(self.device)

            seq0 = rearrange(
                to_canconical_form(P[:, 0])[:, :57], "t jd -> t 1 jd"
            )  # noqa E501
            seq1 = rearrange(
                to_canconical_form(P[:, 1])[:, :57], "t jd -> t 1 jd"
            )  # noqa E501
            seq2 = rearrange(
                to_canconical_form(P[:, 2])[:, :57], "t jd -> t 1 jd"
            )  # noqa E501

            P_canon = torch.cat([seq0, seq1, seq2], dim=1).unsqueeze(0)

            self.model.eval()
            (
                out_final0,
                out_final1,
                out_final2,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self.model(P_canon)

            out_final0 = (F.sigmoid(out_final0) > 0.5) * 1
            out_final1 = (F.sigmoid(out_final1) > 0.5) * 1
            out_final2 = (F.sigmoid(out_final2) > 0.5) * 1

        return rearrange(
            torch.cat([out_final0, out_final1, out_final2], dim=0)
            .cpu()
            .numpy(),  # noqa E501
            "p t -> t p",
        )


class WhoSpeaksDecider:
    def __init__(self, device):
        fname_local = join(os.getcwd(), "models/whotalks.ckpt")
        assert isfile(fname_local)

        self.model = WhoIsTheBuyerModule.load_from_checkpoint(fname_local)
        self.model.eval().to(device)
        self.device = device

    def eval(self, P):
        return self.model.eval_chunk(P)