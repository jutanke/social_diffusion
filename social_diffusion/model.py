import math
import torch
import torch.nn as nn
from social_diffusion.transforms.rotation import tn_rot3d_c
from social_diffusion.diffusion import DiffusionModel
from social_diffusion.skeleton import Skeleton

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TemporalConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=(kernel_size - 1),
            stride=stride,
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        """
        param x: {n_batch x dim x n_frames}
        """
        chomp = self.kernel_size - self.stride
        return self.conv(x)[:, :, :-chomp].contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, time_dim: int = -1):
        super().__init__()
        self.time_dim = time_dim
        if dim_in == dim_out:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv1d(dim_in, dim_out, kernel_size=1)

        if self.time_dim > 0:
            self.mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_dim, dim_out * 2)
            )  # noqa E501

        self.net = nn.Sequential(
            TemporalConv1d(
                dim_in=dim_in, dim_out=dim_out, kernel_size=3, stride=1
            ),  # noqa E501
            nn.SiLU(),
            Rearrange("b d t -> b t d"),
            nn.LayerNorm(dim_out),
            Rearrange("b t d -> b d t"),
            TemporalConv1d(
                dim_in=dim_out, dim_out=dim_out, kernel_size=3, stride=1
            ),  # noqa E501
        )

    def forward(self, x, time_emb=None):
        """
        :param x: {n_batch x dim x n_frames}
        :param time_emb: {n_batch x time_dim}
        """
        out = self.net(x)

        if self.time_dim > 0:
            time_scale, time_shift = self.mlp(time_emb).chunk(2, dim=1)
            time_scale = rearrange(time_scale, "b d -> b d 1")
            time_shift = rearrange(time_shift, "b d -> b d 1")
            out = out * (time_scale + 1) + time_shift

        return out + self.proj(x)


class Down(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim=-1):
        super().__init__()
        self.block1 = ResidualBlock(
            dim_in=dim_out, dim_out=dim_out, time_dim=time_dim
        )  # noqa E501
        self.down = nn.Sequential(
            TemporalConv1d(
                dim_in=dim_in, dim_out=dim_out, kernel_size=3, stride=2
            ),  # noqa E501
        )

    def forward(self, x, time_embed=None):
        x = self.down(x)
        return self.block1(x, time_embed)


class Up(nn.Module):
    def __init__(self, dim_in, dim_out, time_dim=-1):
        super().__init__()
        self.block1 = ResidualBlock(
            dim_in=dim_out, dim_out=dim_out, time_dim=time_dim
        )  # noqa E501
        self.up = nn.Sequential(
            TemporalConv1d(
                dim_in=dim_in, dim_out=dim_out, kernel_size=3, stride=1
            ),  # noqa E501
            nn.Upsample(scale_factor=2, mode="linear"),
        )

    def forward(self, x, time_embed=None):
        x = self.up(x)
        return self.block1(x, time_embed)


def apply_normalization_per_person(Seq, R, T):
    """
    :param Seq: {n_batch x n_frames x n_person x d}
    :param R: {n_batch x n_person x 3 x 3}
    :param T: {n_batch x n_person x 3}
    """
    n_person = Seq.size(2)
    Seq_for_pid = []  # n_batch x n_person x n_person x d
    for pid in range(n_person):
        Seq_other = rearrange(Seq.clone(), "b t p d -> (b p) t d")
        R_pid = repeat(R[:, pid], "b d1 d2 -> (b p) d1 d2", p=n_person)
        T_pid = repeat(T[:, pid], "b d -> (b p) 1 d", p=n_person)
        Seq_other = Seq_other - T_pid
        Seq_other = torch.bmm(Seq_other, R_pid)
        Seq_other = rearrange(Seq_other, "(b p) t d -> b t p d", p=n_person)
        Seq_for_pid.append(Seq_other)
    Seq_for_pid = rearrange(
        torch.stack(Seq_for_pid), "p2 b t p1 d -> b t p1 p2 d"
    )  # noqa E501
    return Seq_for_pid


def undo_normalization_per_person(Seq, R, T):
    """
    :param Seq: {n_batch x n_frames x n_person x d}
    :param R: {n_batch x n_person x 3 x 3}
    :param T: {n_batch x n_person x 3}
    """
    n_person = Seq.size(2)
    Seq = rearrange(Seq, "b t p d -> (b p) t d")
    R_T = rearrange(R, "b p d1 d2 -> (b p) d2 d1")
    T = rearrange(T, "b p d -> (b p) 1 d")
    return rearrange(
        torch.bmm(Seq, R_T) + T, "(b p) t d -> b t p d", p=n_person
    )  # noqa E501


class MotionModel(DiffusionModel):
    def __init__(
        self,
        n_in,
        n_frames,
        skel: Skeleton,
        fourier_dim=64,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__(skel=skel)
        self.n_in = n_in
        pose_dim = self.jd()

        hidden_others = hidden_dim // 4

        time_dim = fourier_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.proj = nn.Conv1d(pose_dim, hidden_dim, kernel_size=1, padding=0)

        self.down1_others = Down(
            dim_in=pose_dim, dim_out=hidden_others, time_dim=time_dim
        )
        self.down2_others = Down(
            dim_in=hidden_others, dim_out=hidden_others, time_dim=time_dim
        )
        self.down3_others = Down(
            dim_in=hidden_others, dim_out=hidden_others, time_dim=time_dim
        )

        self.down1 = Down(
            dim_in=hidden_dim, dim_out=hidden_dim * 2, time_dim=time_dim
        )  # noqa E501
        self.down2 = Down(
            dim_in=hidden_dim * 2, dim_out=hidden_dim * 2, time_dim=time_dim
        )
        self.down3 = Down(
            dim_in=hidden_dim * 2, dim_out=hidden_dim * 2, time_dim=time_dim
        )
        self.res1 = ResidualBlock(
            dim_in=hidden_dim * 2, dim_out=hidden_dim * 2, time_dim=time_dim
        )

        self.up3 = Up(
            dim_in=hidden_dim * 2 + hidden_dim * 2,
            dim_out=hidden_dim * 2,
            time_dim=time_dim,
        )
        self.up2 = Up(
            dim_in=hidden_dim * 2 + hidden_dim * 2,
            dim_out=hidden_dim * 2,
            time_dim=time_dim,
        )
        self.up1 = Up(
            dim_in=hidden_dim * 2 + hidden_dim * 2,
            dim_out=hidden_dim,
            time_dim=time_dim,
        )

        self.final_res = ResidualBlock(
            dim_in=hidden_dim + hidden_dim,
            dim_out=hidden_dim,
            time_dim=time_dim,  # noqa E501
        )

        self.to_pose = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=pose_dim,
                kernel_size=1,
                padding=0,  # noqa E501
            ),
        )

        self.merge_history = nn.Conv1d(
            hidden_dim * 2 + hidden_others,
            hidden_dim * 2,
            kernel_size=1,
            padding=0,  # noqa E501
        )

    def forward(self, X_0, X_t, time, batch):
        """
        :param X_0: {n_batch x n_times x n_person x jd} clean gt image
        :param X_t: {n_batch x n_times x n_person x jd} properly noised
        :param batch: {
            "X_0": ...
            ...
        }
        """
        n_batch = X_t.size(0)
        n_frames = X_t.size(1)
        n_person = X_t.size(2)

        time = repeat(time, "b -> (b p)", p=n_person)
        t = self.time_mlp(time)

        # normalization:

        X_0_rot = X_0[:, :, :, 3:6]
        X_0_loc = X_0[:, :, :, :3]

        T = X_0_loc[:, self.n_in - 1, :, :].clone()
        X_0_rot = X_0_rot[:, self.n_in - 1, :, :]  # b p 3
        # print("newT", T.shape)
        # print("X_0_rot", X_0_rot.shape)

        x = X_0_rot[:, :, 0]
        y = X_0_rot[:, :, 1]
        rads = torch.atan2(y, x)
        rads = rearrange(rads, "b p -> (b p) 1")
        R = tn_rot3d_c(rads)
        R = rearrange(R, "(b p) d1 d2 -> b p d1 d2", p=n_person)

        X_t_pos = X_t[:, :, :, :3]

        X_t_rest = repeat(
            X_t[:, :, :, 3:], "b t p1 d -> b t p1 p2 d", p2=n_person
        )  # noqa E501

        X_t_pos = apply_normalization_per_person(X_t_pos, R, T)  # b t p1 p2 jd

        # rotate rotation points:
        #     * X_t_rot -> b t p 3
        #     * R -> b p 3 3
        X_t_rot = X_t_rest[:, :, :, :, 3:6]
        X_t_rest = X_t_rest[:, :, :, :, 3:]

        R_rot = repeat(R, "b p d1 d2 -> (b p p2) d1 d2", p2=n_person)
        X_t_rot = rearrange(X_t_rot, "b t p1 p2 d -> (b p1 p2) t d")

        X_t_rot = torch.bmm(X_t_rot, R_rot)
        X_t_rot = rearrange(
            X_t_rot,
            "(b p1 p2) t d -> b t p1 p2 d",
            p1=n_person,
            p2=n_person,  # noqa E501
        )
        X_t_all = torch.cat([X_t_pos, X_t_rot, X_t_rest], dim=4)

        X_t = []
        Hist = []
        for pid in range(n_person):
            X_t.append(X_t_all[:, :, pid, pid])

            x = rearrange(X_t_all[:, :, pid], "b t p jd -> (b p) jd t")
            d1 = self.down1_others(x, t)  # n_frames // 2
            d2 = self.down2_others(d1, t)  # n_frames // 4
            d3 = self.down3_others(d2, t)  # n_frames // 8
            d3 = reduce(d3, "(b p) d t -> b d t", "mean", p=n_person)
            Hist.append(d3)

        # 2, 64, 128, 32
        Hist = torch.stack(Hist)  # p2 b d t
        Hist = rearrange(Hist, "p b d t -> (b p) d t")

        # Hist = rearrange(torch.stack(Hist), )
        X_t = rearrange(torch.stack(X_t), "p b t jd -> b t p jd")

        X_t = rearrange(X_t, "b t p jd -> (b p) jd t")

        x = self.proj(X_t)

        d1 = self.down1(x, t)  # n_frames // 2
        d2 = self.down2(d1, t)  # n_frames // 4
        d3 = self.down3(d2, t)  # n_frames // 8

        h1 = self.res1(d3, t)  # n_frames // 8

        h2 = self.merge_history(torch.cat([h1, Hist], dim=1))

        u3 = self.up3(torch.cat([h2, d3], dim=1), t)  # n_frames // 4
        u2 = self.up2(torch.cat([u3, d2], dim=1), t)  # n_frames // 2
        u1 = self.up1(torch.cat([u2, d1], dim=1), t)  # n_frames

        final_inp = torch.cat([u1, x], dim=1)

        pose_out = self.to_pose(self.final_res(final_inp, t))

        pose_out = rearrange(
            pose_out,
            "(b p) jd t -> b t p jd",
            jd=self.jd(),
            p=n_person,
            t=n_frames,
            b=n_batch,
        )

        pose_out_loc = pose_out[:, :, :, :3]
        pose_out_rot = pose_out[:, :, :, 3:6]
        pose_out_rest = pose_out[:, :, :, 6:]

        pose_out_loc = undo_normalization_per_person(pose_out_loc, R=R, T=T)

        R_T = rearrange(R, "b p d1 d2 -> (b p) d2 d1")
        pose_out_rot = rearrange(pose_out_rot, "b t p d -> (b p) t d")
        pose_out_rot = torch.bmm(pose_out_rot, R_T)
        pose_out_rot = rearrange(
            pose_out_rot, "(b p) t d -> b t p d", p=n_person
        )  # noqa E501

        pose_out = torch.cat(
            [pose_out_loc, pose_out_rot, pose_out_rest], dim=3
        )  # noqa E501

        return pose_out