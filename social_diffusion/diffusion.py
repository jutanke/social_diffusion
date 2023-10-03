"""
https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
from collections import namedtuple
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce, repeat, rearrange
from social_diffusion.skeleton import Skeleton
from torch import nn
import numpy.linalg as la
from tqdm import tqdm
from social_diffusion.transforms.rotation import rot3d

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def overwrite(inpainting_mask, P_source, P_target):
    """
    P_source -> P_target

    :param inpainting_mask: {b x t x p x jd}
    :param P_source: {b x t x p x jd}
    :param P_target: {b x t x p x jd}
    """
    assert P_source.shape == P_target.shape
    # assert P_source.shape == inpainting_mask.shape
    b, t, p, jd = P_source.shape

    flat_mask = rearrange(inpainting_mask, "b t p jd -> (b t p jd)")

    indices = np.nonzero(flat_mask)[0]

    P_source_flat = rearrange(P_source, "b t p jd -> (b t p jd)")
    P_target_flat = rearrange(P_target, "b t p jd -> (b t p jd)")

    P_target_flat[indices] = P_source_flat[indices]
    return rearrange(P_target_flat, "(b t p jd) -> b t p jd", b=b, t=t, p=p, jd=jd)


class DiffusionModel(nn.Module):
    """
    Template model for the diffusion process
    """

    def __init__(
        self,
        skel: Skeleton,
    ):
        super().__init__()
        self._skel = skel

    def skel(self):
        return self._skel

    def jd(self):
        return self._skel.n_joints * 3

    def forward(self, X_0, X_t, batch):
        raise NotImplementedError("Nope")


def exists(x):
    return x is not None


def identity(t, *args, **kwargs):
    return t


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def l2norm(t):
    return F.normalize(t, dim=-1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    )  # noqa E501
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: DiffusionModel,
        *,
        n_in: int,
        n_frames: int,
        n_person: int,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="cosine",
        # p2 loss weight, from https://arxiv.org/abs/2204.00227 -
        #    0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
        ddim_sampling_eta=1.0,
        # same_noise_for_all_dims=False,
        # additional_loss_fn=None,
    ):
        super().__init__()
        self.model = model

        # self.same_noise_for_all_dims = same_noise_for_all_dims

        # if additional_loss_fn is None:
        #     self.additional_loss_fn = lambda gt, pred: 0
        # else:
        #     self.additional_loss_fn = additional_loss_fn

        self.n_in = n_in
        assert n_frames > n_in

        self.n_frames = n_frames
        self.n_person = n_person

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
        }, "objective must be either pred_noise (predict noise) or pred_x0(predict image start)"  # noqa E501

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )  # noqa E501
        register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )  # noqa E501
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # noqa E501, above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # noqa E501, below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),  # noqa E501
        )

        # calculate p2 reweighting

        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )  # noqa E501

    def model_predictions(self, x, t, x_0, batch, clip_x_start=False):
        model_output = self.model(X_0=x_0, X_t=x, time=t, batch=batch)
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0)
            if clip_x_start
            else identity  # noqa E501
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    # def p_mean_variance(self, x, t, x_0, batch, clip_denoised=True):
    def p_mean_variance(self, x, t, x_0, batch, clip_denoised=False):
        preds = self.model_predictions(x=x, t=t, x_0=x_0, batch=batch)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.q_posterior(  # noqa E501
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t: int,
        x_0: torch.Tensor,
        batch: torch.Tensor,
        clip_denoised=False,
        init_rand=None,
    ):
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long
        )  # noqa E501

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            x_0=x_0,
            clip_denoised=clip_denoised,
            batch=batch,
        )

        if init_rand is None:
            noise = (
                torch.randn_like(x) if t > 0 else 0.0
            )  # no noise if t == 0   # noqa E501
        else:
            noise = init_rand
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop_with_input(
        self,
        batch,
        return_all_steps=False,
        # add_gt=True,
        add_gt=False,
        linear_blending=None,
        init_rand=None,
        return_noise=False,
        n_diffusion_steps=-1,
        ds_train=None,
    ):
        """
        :param batch: {
            "X_0": ...
        }
        """
        device = self.betas.device

        P_out = batch["X_0"]
        P_out = P_out.to(device)

        Imgs = []

        if init_rand is None:
            img = torch.randn(P_out.shape, device=device)
        else:
            img = init_rand

        init_noise = img.clone()

        img = img.contiguous()

        # img_last = None

        img[:, : self.n_in] = P_out[:, : self.n_in]
        if linear_blending is not None:
            lin_blend_start, lin_blend_end = linear_blending
            img[:, lin_blend_start:lin_blend_end] = P_out[
                :, lin_blend_start:lin_blend_end
            ]

        if return_all_steps:
            Imgs.append(img.clone().cpu().numpy())

        x_start = None

        def sqish(t, num_timesteps, shift=0):
            x = num_timesteps / 2 - t + shift
            return 1 - (1 / (1 + np.exp(-x * 0.01)))

        n_diffusion_steps = (
            n_diffusion_steps if n_diffusion_steps > 0 else self.num_timesteps
        )
        for t in tqdm(
            reversed(range(0, n_diffusion_steps)),
            desc="sampling loop time step",
            total=n_diffusion_steps,
            position=0,
            leave=True,
        ):
            img, x_start = self.p_sample(
                x=img, t=t, x_0=P_out, init_rand=init_rand, batch=batch
            )
            # img_last = img.copy()

            # adjust orientation!
            if ds_train is not None:
                n_person = P_out.size(2)
                n_frames = P_out.size(1)

                img_ = []
                n_samples = img.shape[0]
                for ii in range(n_samples):
                    img_seq = ds_train.undo_standardization(img[ii])

                    p_out_seq = ds_train.undo_standardization(
                        P_out[ii].detach().cpu().numpy()
                    )

                    xy_pred = img_seq[self.n_in - 1, :, 3:5]
                    xy_pred = xy_pred / la.norm(xy_pred, axis=1, keepdims=True)

                    x_pred = xy_pred[:, 0]
                    y_pred = xy_pred[:, 1]
                    rads_pred = np.arctan2(y_pred, x_pred)

                    x_real = p_out_seq[self.n_in - 1, :, 3]
                    y_real = p_out_seq[self.n_in - 1, :, 4]

                    rads_real = np.arctan2(y_real, x_real)

                    rads_pred2real = rads_pred - rads_real

                    for iii in range(n_person):
                        translation = (
                            p_out_seq[self.n_in - 1, iii, :3]
                            - img_seq[self.n_in - 1, iii, :3]
                        )
                        rad = rads_pred2real[iii]
                        if rad < 0:
                            rad += 2 * np.pi
                        R = np.transpose(rot3d(0, 0, rad))

                        for frame in range(n_frames):
                            pred = np.zeros((3,), dtype=np.float32)
                            pred[:2] = img_seq[frame, iii, 3:5]
                            img_seq[frame, iii, 3:6] = R @ pred
                            img_seq[frame, iii, :3] += translation

                    img_seq = ds_train.do_standardization(img_seq)
                    img_.append(img_seq)

                img = torch.from_numpy(np.array(img_)).to(P_out.device)

            if return_all_steps:
                Imgs.append(img.clone().cpu().numpy())
            img[:, : self.n_in] = P_out[:, : self.n_in]

            if linear_blending is not None:
                lin_blend_start, lin_blend_end = linear_blending
                blend_value = np.linspace(
                    start=0.95,
                    stop=0.0001,
                    num=lin_blend_end - lin_blend_start,  # noqa E501
                )
                for i, blend_t in enumerate(
                    range(lin_blend_start, lin_blend_end)
                ):  # noqa E501
                    pred = P_out[:, blend_t]
                    diff = img[:, blend_t]

                    pred_w = blend_value[i]
                    diff_w = 1.0 - pred_w

                    img[:, blend_t] = pred * pred_w + diff * diff_w

            # img[:, lin_blend_start:lin_blend_end] = P_out[
            #     :, lin_blend_start:lin_blend_end
            # ]

        if add_gt:
            img[:, : self.n_in] = P_out[:, : self.n_in]
            if linear_blending is not None:
                lin_blend_start, lin_blend_end = linear_blending
                blend_value = np.linspace(
                    start=0.95,
                    stop=0.0001,
                    num=lin_blend_end - lin_blend_start,  # noqa E501
                )
                for i, blend_t in enumerate(
                    range(lin_blend_start, lin_blend_end)
                ):  # noqa E501
                    pred = P_out[:, blend_t]
                    diff = img[:, blend_t]

                    pred_w = blend_value[i]
                    diff_w = 1.0 - pred_w

                    img[:, blend_t] = pred * pred_w + diff * diff_w
        if return_all_steps:
            assert not return_noise
            Imgs.append(img.cpu().numpy())
            return Imgs

        if return_noise:
            return img, init_noise

        return img

    @torch.no_grad()
    def p_sample_loop(self, shape):
        _, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img, x_start = self.p_sample(img, t, x_0=img, batch={})

        return img

    @torch.no_grad()
    def inpainting_sample_with_input(
        self, batch, *, n_diffusion_steps=1000, ds_train=None
    ):
        return self.p_inpatining_sample_loop_with_input(
            batch=batch, n_diffusion_steps=n_diffusion_steps, ds_train=ds_train
        )

    @torch.no_grad()
    def p_inpatining_sample_loop_with_input(
        self, batch, *, n_diffusion_steps=1000, ds_train=None
    ):
        device = self.betas.device
        P_out = batch["X_0"]
        inpainting_mask = batch["inpainting_mask"]
        P_out = P_out.to(device)
        img = torch.randn(P_out.shape, device=device)
        # def overwrite(inpainting_mask, P_source, P_target):
        img = overwrite(
            inpainting_mask=inpainting_mask, P_source=P_out, P_target=img
        ).contiguous()

        n_diffusion_steps = (
            n_diffusion_steps if n_diffusion_steps > 0 else self.num_timesteps
        )
        for t in tqdm(
            reversed(range(0, n_diffusion_steps)),
            desc="sampling loop time step",
            total=n_diffusion_steps,
            position=0,
            leave=True,
        ):
            img, x_start = self.p_sample(
                x=img, t=t, x_0=P_out, init_rand=None, batch=batch
            )
            img = overwrite(
                inpainting_mask=inpainting_mask, P_source=P_out, P_target=img
            ).contiguous()

        return img.detach().cpu().numpy()

    @torch.no_grad()
    def sample_with_input(
        self,
        batch,
        linear_blending=None,
        return_all_steps=False,
        init_rand=None,
        return_noise=False,
        n_diffusion_steps=-1,
        ds_train=None,
    ):
        return self.p_sample_loop_with_input(
            batch,
            linear_blending=linear_blending,
            return_all_steps=return_all_steps,
            init_rand=init_rand,
            return_noise=return_noise,
            n_diffusion_steps=n_diffusion_steps,  # JULIAN
            add_gt=False,
            ds_train=ds_train,
        )

    @torch.no_grad()
    def sample(self, batch_size=8, n_frames=None):
        if n_frames is None:
            n_frames = self.n_frames
        sample_fn = (
            self.p_sample_loop
            if not self.is_ddim_sampling
            else self.ddim_sample  # noqa E501
        )
        return sample_fn(
            (batch_size, n_frames, self.n_person, self.model.jd())
        )  # noqa E501

    def q_sample(self, x_start, t, noise=None):
        """
        :param x_start: {n_batch x n_frames x n_person x dim}
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  # noqa E501
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, P_out, t, batch, noise=None):
        """
        :param P_out: {n_frames x n_frames x 3 x 57}
        """
        x_start = P_out

        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.model(X_0=x_start, X_t=x, time=t, batch=batch)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = self.loss_fn(model_out, target, reduction="none")
        # loss = reduce(loss, "b d t j -> t", "mean")

        weights_per_j = torch.from_numpy(
            self.model.skel().get_loss_weight()
        ).to(  # noqa E501
            loss.device
        )

        loss = reduce(loss, "b d t j -> j", "mean")
        loss = loss * weights_per_j
        # b t p jd

        loss = loss.mean()

        final_loss = loss
        return final_loss, model_out

    def forward(self, batch, *args, **kwargs):
        """
        :param P_in: {n_batch x n_frames x 3 x 57}
        """
        P_out = batch["X_0"]
        b = P_out.size(0)
        device = P_out.device

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(P_out, t, batch, *args, **kwargs)

    def inpainting(
        self,
        P,
        inpainting_mask,
        ds_train,
        *,
        device=None,
        n_samples=1,
    ):
        """
        :param P:
        """

        skel = self.model.skel()
        if device is None:
            device = torch.device("cuda:0")
        assert isinstance(P, np.ndarray)
        if len(P.shape) != 3 or P.shape[2] != skel.jd_without_normalization():
            raise ValueError(f"Weird shape: {P.shape}")

        n_frames = P.shape[0]
        # if n_in == -1:
        #     n_in = self.n_in
        n_person = P.shape[1]

        P_norm = np.zeros((n_frames, n_person, skel.jd()), dtype=np.float32)
        P_norm, undo_data = ds_train.create_batch_entry(P)

        # if after_create_batch_entry_fn is not None:
        #     P_norm = after_create_batch_entry_fn(P_norm)

        P_norm = ds_train.do_standardization(P_norm)

        # P_norm[n_in:] = 0
        # for t in range(n_in, len(P_norm)):
        #     P_norm[t] = P_norm[n_in - 1]

        P_norm = torch.from_numpy(P_norm).to(device).unsqueeze(0)

        if n_samples > 1:
            P_norm = repeat(P_norm, "b t p jd -> (s b) t p jd", s=n_samples)

        inpainting_mask_full = np.ones(P_norm.shape)
        inpainting_mask_full[:, :, :, 6:] = rearrange(
            inpainting_mask, "t p jd -> 1 t p jd"
        )
        # inpainting_mask_full = torch.from_numpy(inpainting_mask_full).float().to(device)

        O = self.inpainting_sample_with_input(
            batch={"X_0": P_norm, "inpainting_mask": inpainting_mask_full},
            # return_noise=False,
            n_diffusion_steps=1000,
            # return_all_steps=False,
            ds_train=ds_train,
        )

        print("P_norm", P_norm.shape)
        print("O", O.shape)
        for ii in range(len(O)):
            O[ii] = ds_train.undo_standardization(O[ii])

        P_out = []
        for ii in range(len(O)):
            P_out.append(ds_train.undo_batch_entry(O[ii], undo_data))

        P_out = np.array(P_out, dtype=np.float32)

        return P_out

    def predict(
        self,
        P,
        ds_train,
        *,
        device=None,
        n_in=-1,
        n_samples=1,
        after_create_batch_entry_fn=None,
        before_undo_standardization_fn=None,
        before_undo_batch_entry_fn=None,
        return_noise=False,
        n_diffusion_steps=-1,
    ):
        """
        actually predicts a pose!
        :param after_create_batch_entry_fn:
            def after_create_batch_entry_fn(P) -> P
        :param before_undo_standardization_fn:
            def before_undo_standardization_fn(P) -> P
        :param before_undo_batch_entry_fn:
            def before_undo_batch_entry_fn(P) -> P
        """
        skel = self.model.skel()
        if device is None:
            device = torch.device("cuda:0")
        assert isinstance(P, np.ndarray)
        if len(P.shape) != 3 or P.shape[2] != skel.jd_without_normalization():
            raise ValueError(f"Weird shape: {P.shape}")

        n_frames = P.shape[0]
        if n_in == -1:
            n_in = self.n_in
        n_person = P.shape[1]

        P_norm = np.zeros((n_frames, n_person, skel.jd()), dtype=np.float32)
        P_norm[:n_in], undo_data = ds_train.create_batch_entry(P[:n_in])

        if after_create_batch_entry_fn is not None:
            P_norm = after_create_batch_entry_fn(P_norm)

        P_norm = ds_train.do_standardization(P_norm)

        P_norm[n_in:] = 0
        for t in range(n_in, len(P_norm)):
            P_norm[t] = P_norm[n_in - 1]

        P_norm = torch.from_numpy(P_norm).to(device).unsqueeze(0)

        if n_samples > 1:
            P_norm = repeat(P_norm, "b t p jd -> (s b) t p jd", s=n_samples)

        Os = self.sample_with_input(
            batch={"X_0": P_norm},
            return_noise=False,
            n_diffusion_steps=n_diffusion_steps,
            return_all_steps=True,
            ds_train=ds_train,
        )
        O = Os[-1]  # noqa E741

        if before_undo_standardization_fn is not None:
            for ii in range(len(O)):
                O[ii] = before_undo_standardization_fn(O[ii])

        for ii in range(len(O)):
            O[ii] = ds_train.undo_standardization(O[ii])

        if before_undo_batch_entry_fn is not None:
            for ii in range(len(O)):
                O[ii] = before_undo_batch_entry_fn(O[ii])

        P_out = []
        for ii in range(len(O)):
            P_out.append(ds_train.undo_batch_entry(O[ii], undo_data))

        P_out = np.array(P_out, dtype=np.float32)

        return P_out