import hashlib
import struct
from typing import Optional, cast, Any, TypeVar
import logging
import urllib.request
import io
import itertools
import random
from tqdm import tqdm
from functools import reduce
from operator import mul
from torch.nn import functional as F
from torch import Tensor
from torch import nn
import math
from pathlib import Path
import os
import numpy as np
from random import randint
from PIL import Image
from skimage.segmentation import slic, find_boundaries
from skimage.measure import regionprops
import torch
import torchvision.models as models
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
# from simple_lama_inpainting import SimpleLama       # fix   
from .linear_layer import LinearLayer                                # fix
from simple_knn._C import distCUDA2  # type: ignore
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # type: ignore
from .eap import E_attr
from . import dinov2
from .cgnet import Context_Guided_Network, WongiLoss
from .unet import Unet_model
from .deformnet import DeformNetwork
from .config import Config
from .types import (
    Method, 
    MethodInfo, 
    RenderOutput, 
    ModelInfo, 
    camera_model_to_int, 
    Dataset, 
    Cameras, 
    GenericCameras,
    OptimizeEmbeddingOutput,
)
T = TypeVar("T")


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def get_torch_checkpoint_sha(checkpoint_data):
    sha = hashlib.sha256()
    def update(d):
        if type(d).__name__ == "Tensor" or type(d).__name__ == "Parameter":
            sha.update(d.cpu().numpy().tobytes())
        elif isinstance(d, dict):
            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                update(k)
                update(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                update(v)
        elif isinstance(d, (int, float)):
            sha.update(struct.pack("f", d))
        elif isinstance(d, str):
            sha.update(d.encode("utf8"))
        elif d is None:
            sha.update("(None)".encode("utf8"))
        else:
            raise ValueError(f"Unsupported type {type(d)}")
    update(checkpoint_data)
    return sha.hexdigest()


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


def camera_project(cameras: GenericCameras[Tensor], xyz: Tensor) -> Tensor:
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    origins = cameras.poses[..., :3, 3]
    rotation = cameras.poses[..., :3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_models, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx, fy, cx, cy = torch.moveaxis(cameras.intrinsics, -1, 0)
    x = fx * x + cx
    y = fy * y + cy
    return torch.stack((x, y), -1)


def safe_state():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


def ssim_down(x, y, max_size=None):
    osize = x.shape[2:]
    if max_size is not None:
        scale_factor = max(max_size/x.shape[-2], max_size/x.shape[-1])
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')
    out = ssim(x, y, size_average=False).unsqueeze(1)
    if max_size is not None:
        out = F.interpolate(out, size=osize, mode='bilinear', align_corners=False)
    return out.squeeze(1)


def _ssim_parts(img1, img2, window_size=11):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def msssim(x, y, max_size=None, min_size=200):
    raw_orig_size = x.shape[-2:]
    if max_size is not None:
        scale_factor = min(1, max(max_size/x.shape[-2], max_size/x.shape[-1]))
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')

    ssim_maps = list(_ssim_parts(x, y))
    orig_size = x.shape[-2:]
    while x.shape[-2] > min_size and x.shape[-1] > min_size:
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        ssim_maps.extend(tuple(F.interpolate(x, size=orig_size, mode='bilinear') for x in _ssim_parts(x, y)[1:]))
    out = torch.stack(ssim_maps, -1).prod(-1)
    if max_size is not None:
        out = F.interpolate(out, size=raw_orig_size, mode='bilinear')
    return out.mean(1)


def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = ((nh + 13) // 14) * 14
        nw = ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode='bilinear')
    return x


# class DiscriminatorModel(nn.Module):
#     self.

class UncertaintyModel(nn.Module):
    img_norm_mean: Tensor
    img_norm_std: Tensor

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = getattr(dinov2, config.uncertainty_backbone)(pretrained=True)
        self.patch_size = self.backbone.patch_size
        in_features = self.backbone.embed_dim
        self.conv_seg = nn.Conv2d(in_features, 1, kernel_size=1)
        self.bn = nn.SyncBatchNorm(in_features)
        nn.init.normal_(self.conv_seg.weight.data, 0, 0.01)
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)

        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean / 255.)
        self.register_buffer("img_norm_std", img_norm_std / 255.)

        self._images_cache = {}

        # Freeze dinov2 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _initialize_head_from_checkpoint(self):
        # ADA20 classes to ignore
        cls_to_ignore = [13, 21, 81, 84]
        # Pull the checkpoint
        backbone = self.config.uncertainty_backbone
        url = f"https://dl.fbaipublicfiles.com/dinov2/{backbone}/{backbone}_ade20k_linear_head.pth"
        with urllib.request.urlopen(url) as f:
            checkpoint_data = f.read()
        checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
        old_weight = checkpoint["state_dict"]["decode_head.conv_seg.weight"]
        new_weight = torch.empty(1, old_weight.shape[1], 1, 1)
        nn.init.normal_(new_weight, 0, 0.0001)
        new_weight[:, cls_to_ignore] = old_weight[:, cls_to_ignore] * 1000
        nn.init.zeros_(assert_not_none(self.conv_seg.bias).data)
        self.conv_seg.weight.data.copy_(new_weight)

        # Load the bn data
        self.bn.load_state_dict({k[len("decode_head.bn."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("decode_head.bn.")})

    def _get_dino_cached(self, x, cache_entry=None):
        if cache_entry is None or (cache_entry, x.shape) not in self._images_cache:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(x, n=[self.backbone.num_heads-1], reshape=True)[-1]
            if cache_entry is not None:
                self._images_cache[(cache_entry, x.shape)] = x.detach().cpu()
        else:
            x = self._images_cache[(cache_entry, x.shape)].to(x.device)
        return x

    def _compute_cosine_similarity(self, x, y, _x_cache=None, _y_cache=None, max_size=None):
        # Normalize data
        h, w = x.shape[2:]
        if max_size is not None and (max_size < h or max_size < w):
            assert max_size % 14 == 0, "max_size must be divisible by 14"
            scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
            nh = int(h * scale_factor)
            nw = int(w * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
            y = F.interpolate(y, size=(nh, nw), mode='bilinear')

        x = (x - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        y = (y - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        padded_shape = x.shape
        y = F.pad(y, pads)

        with torch.no_grad():
            x = self._get_dino_cached(x, _x_cache)
            y = self._get_dino_cached(y, _y_cache)

        cosine = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        cosine: Tensor = F.interpolate(cosine, size=padded_shape[2:], mode="bilinear", align_corners=False)
        
        # Remove padding
        cosine = cosine[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        if max_size is not None and (max_size < h or max_size < w):
            cosine = F.interpolate(cosine, size=(h, w), mode='bilinear', align_corners=False)
        return cosine.squeeze(1)
    
    def _forward_uncertainty_features(self, inputs: Tensor, _cache_entry=None) -> Tensor:
        # Normalize data
        inputs = (inputs - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        h, w = inputs.shape[2:]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in inputs.shape[:1:-1]))
        inputs = F.pad(inputs, pads)

        x = self._get_dino_cached(inputs, _cache_entry)

        x = F.dropout2d(x, p=self.config.uncertainty_dropout, training=self.training)
        x = self.bn(x)
        logits = self.conv_seg(x)
        # We could also do this using weight init, 
        # but we want to have a prior then doing L2 regularization
        logits = logits + math.log(math.exp(1) - 1)

        # Rescale to input size
        logits = F.softplus(logits)
        logits: Tensor = F.interpolate(logits, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        logits = logits.clamp(min=self.config.uncertainty_clip_min)

        # Add padding
        logits = logits[:, :, pads[2]:h+pads[2], pads[0]:w+pads[0]]
        return logits

    @property
    def device(self):
        return self.img_norm_mean.device

    def forward(self, image: Tensor, _cache_entry=None):
        return self._forward_uncertainty_features(image, _cache_entry=_cache_entry)

    def setup_data(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def _load_image(self, img):
        return torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)[None]).to(self.device)

    def _scale_input(self, x, max_size: Optional[int] = 504):
        h, w = nh, nw = x.shape[2:]
        if max_size is not None:
            scale_factor = min(max_size/x.shape[-2], max_size/x.shape[-1])
            if scale_factor >= 1:
                return x
            nw = int(w * scale_factor)
            nh = int(h * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
        return x

    def _dino_plus_ssim(self, gt, prediction, _cache_entry=None, max_size=None):
        gt_down = dino_downsample(gt, max_size=max_size)
        prediction_down = dino_downsample(prediction, max_size=max_size)
        dino_cosine, cosine_map = self._compute_cosine_similarity(
            gt_down,
            prediction_down,
            _x_cache=_cache_entry).detach()
        dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
        msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
        return torch.min(dino_part, msssim_part), cosine_map

    def _compute_losses(self, gt, prediction, iteration, prefix='', _cache_entry=None):
        uncertainty = self(self._scale_input(gt, self.config.uncertainty_dino_max_size), _cache_entry=_cache_entry)
        log_uncertainty = torch.log(uncertainty)
        # _dssim_go = dssim_go(gt, prediction, size_average=False).unsqueeze(1).clamp_max(self.config.uncertainty_dssim_clip_max)
        # _dssim_go = 1 - ssim(gt, prediction).unsqueeze(1)
        _ssim = ssim_down(gt, prediction, max_size=400).unsqueeze(1)
        # if iteration > 29500:
        #     tensor_to_uncertainty_img(_ssim.squeeze(0).squeeze(0).detach().cpu().numpy(), iteration, _cache_entry[1], "ssim")
        _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)
        # if iteration > 29500:
        #     tensor_to_uncertainty_img(_ssim.squeeze(0).squeeze(0).detach().cpu().numpy(), iteration, _cache_entry[1], "_msssim")

        if self.config.uncertainty_mode == "l2reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / (2 * uncertainty.pow(2))
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "l1reg":
            if uncertainty.shape[2:] != gt.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = 1 / uncertainty
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult
        elif self.config.uncertainty_mode == "dino":
            # loss_mult = 1 / (2 * uncertainty.pow(2))
            # loss_mult = 1 / uncertainty
            # Compute dino loss
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            dino_cosine, cosine_map = self._compute_cosine_similarity(
                gt_down,
                prediction_down,
                _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            uncertainty_loss = dino_part * dino_downsample(loss_mult, max_size=350)
            if iteration > 195000:       # fix
                tensor_to_uncertainty_img(uncertainty_loss.squeeze(0).squeeze(0).detach().cpu().numpy(), iteration, _cache_entry[1], "uncertainty")
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)

        elif self.config.uncertainty_mode == "dino+mssim":
            loss_mult = 1 / (2 * uncertainty.pow(2))
            gt_down = dino_downsample(gt, max_size=350)
            prediction_down = dino_downsample(prediction, max_size=350)
            dino_cosine = self._compute_cosine_similarity(
                gt_down,
                prediction_down,
                _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            msssim_part = 1 - msssim(gt_down, prediction_down, min_size=80).unsqueeze(1).detach()
            uncertainty_loss = torch.min(dino_part, msssim_part) * dino_downsample(loss_mult, max_size=350)
            if loss_mult.shape[2:] != gt.shape[2:]:
                loss_mult = F.interpolate(loss_mult, size=gt.shape[2:], mode='bilinear', align_corners=False)
            loss_mult = loss_mult.clamp_max(3)

        else:
            raise ValueError(f"Invalid uncertainty_mode: {self.config.uncertainty_mode}")

        beta = log_uncertainty.mean()
        loss = uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta

        ssim_discounted = (_ssim * loss_mult).sum() / loss_mult.sum()
        mse = torch.pow(gt - prediction, 2)
        mse_discounted = (mse * loss_mult).sum() / loss_mult.sum()
        psnr_discounted = 10 * torch.log10(1 / mse_discounted)

        metrics = {
            f"{prefix}loss": loss.item(),
            f"{prefix}ssim": _ssim.mean().item(),
            f"{prefix}msssim": _msssim.mean().item(),
            f"{prefix}ssim_discounted": ssim_discounted.item(),
            f"{prefix}mse_discounted": mse_discounted.item(),
            f"{prefix}psnr_discounted": psnr_discounted.item(),
            f"{prefix}beta": beta.item(),
        }
        return loss, metrics, loss_mult.detach(), cosine_map

    def get_loss(self, gt_image, image, iteration, prefix='', _cache_entry=None):
        gt_torch = gt_image.unsqueeze(0)
        image = image.unsqueeze(0)
        loss, metrics, loss_mult = self._compute_losses(gt_torch, image, iteration, prefix, _cache_entry=_cache_entry)
        loss_mult = loss_mult.squeeze(0)
        metrics[f"{prefix}uncertainty_loss"] = metrics.pop(f"{prefix}loss")
        metrics.pop(f"{prefix}ssim")
        return loss, metrics, loss_mult

    @staticmethod
    def load(path: str):
        ckpt = torch.load(os.path.join(path, "checkpoint.pth"), map_location="cpu")
        config = OmegaConf.structured(Config)
        config = cast(Config, OmegaConf.merge(config, OmegaConf.create(ckpt.pop("config"))))
        model = UncertaintyModel(config)
        model.load_state_dict(ckpt, strict=False)
        return model

    def save(self, path: str):
        state = self.state_dict()
        state["config"] = OmegaConf.to_yaml(self.config, resolve=True)
        torch.save(state, os.path.join(path, "checkpoint.pth"))


#
# SH eval
#
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getWorld2View2(R, t, translate, scale):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    # P[0, 2] = (w - 2.0 * cx) / w
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_rotation(r, device):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)


# SSIM
def dssim_go(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)

    # Normal dssim would use this
    # dssim_map = (1 - luminance * contrast * structure) / 2
    # NeRF on the Go uses this:
    dssim_map = (1 - luminance) * (1 - contrast) * (1 - structure)

    if size_average:
        return dssim_map.mean()
    else:
        return dssim_map.mean(-3)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def get_uniform_points_on_sphere_fibonacci(num_points, *, dtype=None, xnp=torch):
    # https://arxiv.org/pdf/0912.4540.pdf
    # Golden angle in radians
    if dtype is None:
        dtype = xnp.float32
    phi = math.pi * (3. - math.sqrt(5.))
    N = (num_points - 1) / 2
    i = xnp.linspace(-N, N, num_points, dtype=dtype)
    lat = xnp.arcsin(2.0 * i / (2*N+1))
    lon = phi * i

    # Spherical to cartesian
    x = xnp.cos(lon) * xnp.cos(lat)
    y = xnp.sin(lon) * xnp.cos(lat)
    z = xnp.sin(lat)
    return xnp.stack([x, y, z], -1)


@torch.no_grad()
def get_sky_points(num_points, points3D: Tensor, cameras: GenericCameras[Tensor]):
    xnp = torch
    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=xnp)
    points = points.to(points3D.device)
    mean = points3D.mean(0)[None]
    sky_distance = xnp.quantile(xnp.linalg.norm(points3D - mean, 2, -1), 0.97) * 10
    points = points * sky_distance
    points = points + mean
    gmask = torch.zeros((points.shape[0],), dtype=xnp.bool, device=points.device)
    for cam in tqdm(cameras, desc="Generating skybox"):
        uv = camera_project(cam, points[xnp.logical_not(gmask)])
        mask = xnp.logical_not(xnp.isnan(uv).any(-1))
        # Only top 2/3 of the image
        assert cam.image_sizes is not None
        mask = xnp.logical_and(mask, uv[..., -1] < 2/3 * cam.image_sizes[..., 1])
        gmask[xnp.logical_not(gmask)] = xnp.logical_or(gmask[xnp.logical_not(gmask)], mask)
    return points[gmask], sky_distance / 2


def add_fourier_features(features: torch.Tensor, scale=(0.0, 1.0), num_frequencies=3):
    features = (features - scale[0]) / (scale[1] - scale[0])
    freqs = 2**torch.linspace(0, num_frequencies-1, num_frequencies, dtype=features.dtype, device=features.device)
    offsets = torch.tensor([0, 0.5 * math.pi], dtype=features.dtype, device=features.device)
    sin_cos_features = torch.sin((2*math.pi * (freqs[..., None, :] * features[..., None]).view(*freqs.shape[:-1], -1)).unsqueeze(-1).add(offsets)).view(*features.shape[:-1], -1)
    return torch.cat((features, sin_cos_features), -1)


def srgb_to_linear(img):
    limit = 0.04045

    # NOTE: torch.where is not differentiable, so we use the following
    # return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

    mask = img > limit
    out = img / 12.92
    out[mask] = torch.pow((img[mask] + 0.055) / 1.055, 2.4)
    return out


def linear_to_srgb(img):
    limit = 0.0031308

    # NOTE: torch.where is not differentiable, so we use the following
    # return torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

    mask = img > limit
    out = 12.92 * img
    out[mask] = 1.055 * torch.pow(img[mask], 1.0 / 2.4) - 0.055
    return out


def get_cameras_extent(cameras: Cameras):
    c2w = cameras.poses
    cam_centers = c2w[:, :3, 3:4]
    cam_centers = np.hstack(list(cam_centers))
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    # center = center.flatten()
    radius = diagonal * 1.1
    # translate = -center
    return radius


class ResidualBlock(nn.Sequential):
    def forward(self, input):
        x = super().forward(input)
        minch = min(x.size(1), input.size(1))
        return input[:, :minch] + x[:, :minch]


class AppearanceModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        in_ch = self.config.appearance_embedding_dim + 3
        padding_mode = "reflect"
        self.base = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, padding=0, padding_mode=padding_mode),
            nn.ReLU(),
        )
        ch = 256
        for _ in range(4):
            ch //= 2
            self.base.extend((
                ResidualBlock(
                    nn.Conv2d(ch*2, ch, 3, padding=1, padding_mode=padding_mode),
                    nn.ReLU(),
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
            ))
        self.post_process = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode=padding_mode),
            nn.ReLU(),
            nn.Conv2d(ch, 3, 3, padding=1, padding_mode=padding_mode),
        )
        assert ch == 16

    def forward(self, image, embedding):
        ndim = len(image.shape)
        assert ndim in (3, 4)
        # embedding = F.dropout(embedding, p=0.2, training=self.training)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            embedding = embedding.unsqueeze(0)
        h, w = image.shape[-2:]
        image_down = F.interpolate(image, size=(h//32, w//32), antialias=True, mode="bilinear")
        embedding = embedding[:, :, None, None].repeat(1, 1, h//32, w//32)
        x = torch.cat((image_down, embedding), dim=-3)
        x = self.base(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear")
        x = self.post_process(x)
        x = x * image
        if ndim == 3:
            x = x.squeeze(0)
        return x


def _get_fourier_features(xyz: Tensor, num_features=3):
    xyz = torch.from_numpy(xyz).to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2**torch.linspace(0, num_features-1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    feat = torch.sin(feat).view(-1, reduce(mul, feat.shape[1:]))
    return feat


class EmbeddingModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # sh_coeffs = 4**2
        feat_in = 3
        # print("config.appearance_model_sh: ", config.appearance_model_sh) 
        if config.appearance_model_sh:
            feat_in = ((config.sh_degree + 1) ** 2) * 3
        # print("size: ", feat_in)
        self.mlp = nn.Sequential(
            nn.Linear(config.appearance_embedding_dim + feat_in + 6 * self.config.appearance_n_fourier_freqs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feat_in*2),
        )

    def forward(self, gembedding, aembedding, color, viewdir=None):
        del viewdir  # Viewdirs interface is kept to be compatible with prev. version

        input_color = color
        if not self.config.appearance_model_sh:
            color = color[..., :3]
        inp = torch.cat((color, gembedding, aembedding), dim=-1)
        offset, mul = torch.split(self.mlp(inp) * 0.01, [color.shape[-1], color.shape[-1]], dim=-1)
        offset = torch.cat((offset / C0, torch.zeros_like(input_color[..., offset.shape[-1]:])), dim=-1)
        mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])
        return input_color * mul + offset



class GaussianModel(nn.Module):
    xyz: nn.Parameter
    features_dc: nn.Parameter
    scales: nn.Parameter
    rotations: nn.Parameter
    opacities: nn.Parameter
    xyz_grad: Tensor
    filter_3D: Tensor
    denom: Tensor
    max_radii2D: Tensor

    features_rest: Optional[nn.Parameter]
    embeddings: Optional[nn.Parameter]
    appearance_embeddings: Optional[nn.Parameter]
    appearance_mlp: Optional[nn.Module]

    spatial_lr_scale: Tensor
    active_sh_degree: Tensor

    # Setup functions
    scaling_activation = staticmethod(torch.exp)
    scaling_inverse_activation = staticmethod(torch.log)
    opacity_activation = staticmethod(torch.sigmoid)
    transient_opacity_activation = staticmethod(torch.sigmoid)
    inverse_opacity_activation = staticmethod(torch.special.logit)
    rotation_activation = staticmethod(torch.nn.functional.normalize)

    def __init__(self, 
                 config: Config, 
                 dataset_type,
                 training_setup: bool = True,
                 transient_type = False):
        super().__init__()
        self.optimizer = None
        self._optimizer_state = None
        self.config = config
        self.dataset_type = dataset_type
        self.transient_type = transient_type
        
        if self.transient_type:
            self.implict_mask = Context_Guided_Network(classes=1, M= 2, N=2, input_channel=3)
            self.deform_net = DeformNetwork()
        
        self.register_parameter("xyz", cast(nn.Parameter, nn.Parameter(torch.empty(0, 3, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("features_dc", cast(nn.Parameter, nn.Parameter(torch.empty(0, 3, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("scales", cast(nn.Parameter, nn.Parameter(torch.empty(0, 3, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("rotations", cast(nn.Parameter, nn.Parameter(torch.empty(0, 4, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("opacities", cast(nn.Parameter, nn.Parameter(torch.empty(0, 1, dtype=torch.float32, requires_grad=True))))
        
        if transient_type == False:     # confidence 에 따른
            self.register_parameter("static_confidence", cast(nn.Parameter, nn.Parameter(torch.empty(0, 1, dtype=torch.float32, requires_grad=True))))
        
        self.register_buffer("max_radii2D", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("denom", torch.zeros(0, 1, dtype=torch.float32))
        self.register_buffer("xyz_grad", torch.zeros(0, 1, dtype=torch.float32))
        self.register_buffer("filter_3D", torch.zeros(0, 1, dtype=torch.float32))
        self._dynamically_sized_props = ["xyz", "features_dc", "features_rest", "scales", "rotations", "opacities", "xyz_grad", 
                                         "denom", "filter_3D", "embeddings", "max_radii2D", "static_confidence"]
        if self.config.use_gof_abs_gradient:
            self.register_buffer("xyz_gradient_accum_abs", torch.zeros(0, 1, dtype=torch.float32))
            self.register_buffer("xyz_gradient_accum_abs_max", torch.zeros(0, 1, dtype=torch.float32))
            self._dynamically_sized_props.extend(["xyz_gradient_accum_abs", "xyz_gradient_accum_abs_max"])
        else:
            self.xyz_gradient_accum_abs = None
            self.xyz_gradient_accum_abs_max = None

        self.register_parameter("appearance_embeddings", cast(nn.Parameter, nn.Parameter(torch.empty(0, config.appearance_embedding_dim, dtype=torch.float32, requires_grad=True))))

        if self.config.appearance_enabled:
            self.register_parameter("embeddings", cast(nn.Parameter, nn.Parameter(torch.empty(0, 6*self.config.appearance_n_fourier_freqs, dtype=torch.float32, requires_grad=True))))
            self.appearance_mlp = EmbeddingModel(config)
        else:
            self.register_parameter("embeddings", cast(nn.Parameter, nn.Parameter(torch.empty(0, 6*self.config.appearance_n_fourier_freqs, dtype=torch.float32, requires_grad=True))))
            self.appearance_mlp = EmbeddingModel(config)
        if self.config.sh_degree > 0:
            self.register_parameter("features_rest", cast(nn.Parameter, nn.Parameter(torch.zeros(0, ((self.config.sh_degree + 1) ** 2-1) * 3, dtype=torch.float32, requires_grad=True))))
        else:
            self.features_rest = None
        self.register_buffer("active_sh_degree", torch.full((), 0, dtype=torch.int32))
        self.register_buffer("spatial_lr_scale", torch.zeros((), dtype=torch.float32))

        # Test if checkpoint created by the training code does not use features not implemented in public version
        assert getattr(self.config, "appearance_model_2D", "disabled") == "disabled", "2D appearance models are not supported"
        assert getattr(self.config, "use_background_model", False) is False, "Background model is not supported"
        assert getattr(self.config, "uncertainty_preserve_sky", False) is False, "Flag uncertainty_preserve_sky is not supported"

        if self.config.uncertainty_mode != "disabled":
            self.uncertainty_model = UncertaintyModel(self.config)
        else:
            self.uncertainty_model = None

        if training_setup:
            self.train()
            self._setup_optimizers()
        else:
            self.eval()

    def initialize_from_points3D(self, xyz, colors, spatial_lr_scale : float, opacities=None, static_confidence = None):
        device = self.xyz.device
        self.spatial_lr_scale.fill_(spatial_lr_scale)
        fused_point_cloud = torch.tensor(np.asarray(xyz)).float().to(device)
        # fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().div(255.).cuda())
        fused_color = torch.tensor(np.asarray(colors)).to(self.features_dc.dtype).div(255.).to(device)
        # fused_color = srgb_to_linear(fused_color)

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz)).float().to(device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        if opacities is None:
            opacities = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device)
        else:
            assert len(opacities) == fused_point_cloud.shape[0]
            assert len(opacities.shape) == 1
            opacities = torch.tensor(np.asarray(opacities)).float().to(device).unsqueeze(-1)
        opacities = torch.special.logit(opacities)            
        if self.transient_type == False:     # confidence 에 따른
            if static_confidence is None:
                static_confidence = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device)
            else:
                assert len(static_confidence) == fused_point_cloud.shape[0]
                assert len(static_confidence.shape) == 1
                static_confidence = torch.tensor(np.asarray(static_confidence)).float().to(device).unsqueeze(-1)
            static_confidence = torch.special.logit(static_confidence)
        
        self._resize_parameters(fused_point_cloud.shape[0])
        self.xyz.data.copy_(fused_point_cloud)
        self.features_dc.data.copy_(fused_color)
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.scales.data.copy_(scales)
        self.rotations.data.copy_(rots)
        self.opacities.data.copy_(opacities)
        if self.transient_type == False:     # confidence 에 따른
            self.static_confidence.data.copy_(static_confidence)

        if self.embeddings is not None:
            embeddings = _get_fourier_features(xyz, num_features=self.config.appearance_n_fourier_freqs)
            embeddings.add_(torch.randn_like(embeddings) * 0.0001)
            if not self.config.appearance_init_fourier:
                embeddings.normal_(0, 0.01)
            self.embeddings.data.copy_(embeddings)
        self.max_radii2D.fill_(0.0)

    def _setup_optimizers(self):
        config = self.config
        spatial_lr_scale = self.spatial_lr_scale.cpu().item()
        l = [
            {'params': [self.xyz], 'lr': config.position_lr_init * spatial_lr_scale, "name": "xyz"},
            {'params': [self.features_dc], 'lr': config.feature_lr, "name": "features_dc"},
            {'params': [self.opacities], 'lr': config.opacity_lr, "name": "opacities"},
            {'params': [self.scales], 'lr': config.scaling_lr, "name": "scales"},
            {'params': [self.rotations], 'lr': config.rotation_lr, "name": "rotations"},
        ]
        
        if self.transient_type:
            l.append({'params': self.implict_mask.parameters(), 'lr': 1e-3, "name": "mask_generator"})
            l.append({'params': self.deform_net.parameters(), 'lr': 0.00016 * 5, "name": "deformnet"})
        if self.transient_type == False:     # confidence 에 따른
            l.append({'params': [self.static_confidence], 'lr': config.opacity_lr, "name": "static_confidence"})
        if self.appearance_embeddings is not None:
            l.append({'params': [self.appearance_embeddings], 'lr': config.appearance_embedding_lr, "name": "appearance_embeddings", "weight_decay": config.appearance_embedding_regularization})
        if self.embeddings is not None:
            l.append({'params': [self.embeddings], 'lr': config.embedding_lr, "name": "embeddings"})
        if self.features_rest is not None:
            l.append({'params': [self.features_rest], 'lr': config.feature_lr / 20.0, "name": "features_rest"})
        if self.appearance_mlp is not None:
            l.append({'params': list(self.appearance_mlp.parameters()), 'lr': config.appearance_mlp_lr, "name": "appearance_mlp"})
        if self.uncertainty_model is not None:
            l.append({'params': list(self.uncertainty_model.parameters()), 'lr': config.uncertainty_lr, "name": "uncertainty_model"})
        self.optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=config.position_lr_init*spatial_lr_scale,
                                                    lr_final=config.position_lr_final*spatial_lr_scale,
                                                    lr_delay_mult=config.position_lr_delay_mult,
                                                    max_steps=config.position_lr_max_steps)

    def set_num_training_images(self, num_images):
        if self.appearance_embeddings is not None:
            self._resize_parameter("appearance_embeddings", (num_images, self.appearance_embeddings.shape[1]))
            self.appearance_embeddings.data.normal_(0, 0.01)

    def get_gaussians(self):
        rotations = self.rotation_activation(self.rotations)
        features = self.features_dc
        if self.features_rest is not None:
            features = torch.cat((features, self.features_rest), dim=-1)

        scales = raw_scales = self.scaling_activation(self.scales)
        opacities = self.opacity_activation(self.opacities)
        
        if self.transient_type == False:
            static_confidence = self.transient_opacity_activation(self.static_confidence)
        else:
            static_confidence = None
            
        

        # Apply 3D filter to scales
        scales = (torch.square(raw_scales) + torch.square(self.filter_3D)).sqrt_()

        # apply 3D filter to opacities
        scales_square = torch.square(raw_scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities = opacities * coef[..., None]

        return {
            "xyz": self.xyz,
            "opacities": opacities,
            "scales": scales,
            "rotations": rotations,
            "features": features,
            "static_confidence": static_confidence
        }

    def _resize_parameter(self, name, shape):
        tensor = getattr(self, name)
        new_tensor = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
        new_tensor[:tensor.shape[0]] = tensor
        if isinstance(tensor, nn.Parameter):
            new_param = nn.Parameter(new_tensor.requires_grad_(True))
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    if group["name"] == name:
                        stored_state = self.optimizer.state.get(group['params'][0], None)
                        if stored_state is not None:
                            stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                            stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
                            del self.optimizer.state[group['params'][0]]
                            self.optimizer.state[new_param] = stored_state  # type: ignore
                        group["params"][0] = new_param
                        break
                else:
                    raise ValueError(f"Parameter {name} not found in optimizer")
            setattr(self, name, new_param)
        else:
            self.register_buffer(name, new_tensor)

    def _resize_parameters(self, num_points):
        for name in self._dynamically_sized_props:
            if getattr(self, name, None) is None:
                continue
            self._resize_parameter(name, (num_points, *getattr(self, name).shape[1:]))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Resize all buffers to match the new state_dict
        self._resize_parameters(state_dict["xyz"].shape[0])
        if self.appearance_embeddings is not None:
            self._resize_parameter("appearance_embeddings", state_dict["appearance_embeddings"].shape)
        optimizer = state_dict.pop("optimizer")
        if strict and optimizer is None:
            missing_keys.append("optimizer")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer)
        else:
            self._optimizer_state = optimizer

    def state_dict(self, *, destination: Any = None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.optimizer is None:
            state["optimizer"] = self._optimizer_state
        else:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    @torch.no_grad()
    def compute_3D_filter(self, cameras: Cameras):
        xyz = self.xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:
            assert camera.image_sizes is not None, "Camera image size is not set"
            fx, fy, _, _ = camera.intrinsics
            width, height = camera.image_sizes

            pose = np.copy(camera.poses)
            pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
            pose = np.linalg.inv(pose)
            R = pose[:3, :3]
            T = pose[:3, 3]
            R = np.transpose(R)

            # transform points to camera space
            R = torch.tensor(R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * fx + width / 2.0
            y = y / z * fy + height / 2.0
            
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * width, x <= width * 1.15), torch.logical_and(y >= -0.15 * height, y <= 1.15 * height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < fx:
                focal_length = fx
        
        
        if valid_points.any():
            distance[~valid_points] = distance[valid_points].max()
        else:
            distance[~valid_points] = 0.0
        
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        filter_3D = filter_3D.to(dtype=self.filter_3D.dtype, device=self.filter_3D.device)
        del self.filter_3D
        self.register_buffer("filter_3D", filter_3D[..., None])

    def get_embedding(self, dataset_type, train_image_id=None):
        if train_image_id is not None:
            return self.appearance_embeddings[train_image_id]
        return torch.zeros_like(self.appearance_embeddings[0])

    def oneupSHdegree(self):
        if self.active_sh_degree.cpu().item() < self.config.sh_degree:
            self.active_sh_degree.add_(1)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        assert self.optimizer is not None, "Not set up for training"
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_ply(self, path):
        def construct_list_of_attributes(exclude_filter=True):
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(self.features_dc.shape[1]):
                l.append('f_dc_{}'.format(i))
            # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            #     l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self.scales.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self.rotations.shape[1]):
                l.append('rot_{}'.format(i))
            if not exclude_filter:
                l.append('filter_3D')
            return l

        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.features_dc.detach().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # fuse opacity and scale
        gaussians = self.get_gaussians()
        opacities = torch.special.logit(gaussians["opacities"]).detach().cpu().numpy()
        
        scale = self.scaling_inverse_activation(gaussians["scales"]).detach().cpu().numpy()
        rotation = self.rotations.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        assert self.optimizer is not None, "Not set up for training"
        # reset opacity to by considering 3D filter
        gaussians = self.get_gaussians()
        current_opacity_with_filter = gaussians["opacities"]
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = gaussians["scales"]
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = torch.special.logit(opacities_new)

        # Replace tensors in the optimizer
        for group in self.optimizer.param_groups:
            if group["name"] == "opacities":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(opacities_new)
                stored_state["exp_avg_sq"] = torch.zeros_like(opacities_new)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(opacities_new.requires_grad_(True))
                self.register_parameter("opacities", group["params"][0])
                self.optimizer.state[group['params'][0]] = stored_state

    def _prune_points(self, mask):
        assert self.optimizer is not None, "Not set up for training"
        valid_points_mask = ~mask

        # Prune optimizer
        all_dynamically_sized_props = set(self._dynamically_sized_props)
        for group in self.optimizer.param_groups:
            if group["name"] not in self._dynamically_sized_props:
                continue
            all_dynamically_sized_props.remove(group["name"])
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][valid_points_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                param = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][valid_points_mask].requires_grad_(True))
                param = group["params"][0]
            setattr(self, group["name"], param)

        for name in all_dynamically_sized_props:
            tensor = getattr(self, name, None)
            if tensor is None:
                continue
            self.register_buffer(name, tensor[valid_points_mask])

    def _densification_postfix(self, tensors_dict):
        assert self.optimizer is not None, "Optimizer is not set"
        all_dynamically_sized_props = set(self._dynamically_sized_props)
        for group in self.optimizer.param_groups:
            if group["name"] not in self._dynamically_sized_props:
                continue
            all_dynamically_sized_props.remove(group["name"])
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict.pop(group["name"])
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                parameter = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                parameter = group["params"][0]
            self.register_parameter(group["name"], parameter)

        num_points = len(self.xyz)
        for name in all_dynamically_sized_props:
            tensor = getattr(self, name, None)
            if tensor is None:
                continue
            if name in tensors_dict:
                extension_tensor = tensors_dict.pop(name)
            else:
                extension_tensor = torch.zeros((num_points-len(tensor), *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat((tensor, extension_tensor), dim=0)
            self.register_buffer(name, tensor)

    def _densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        gaussians = self.get_gaussians()
        device = gaussians["xyz"].device
        scales = self.scaling_activation(self.scales)

        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if self.config.use_gof_abs_gradient:
            assert grads_abs is not None
            assert grad_abs_threshold is not None
            padded_grad_abs = torch.zeros((n_init_points), device="cuda")
            padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
            selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > self.config.percent_dense*scene_extent)

        stds = scales[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussians["rotations"][selected_pts_mask], device=device).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians["xyz"][selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(scales[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.rotations[selected_pts_mask].repeat(N,1)
        new_features_dc = self.features_dc[selected_pts_mask].repeat(N,1)
        new_opacity = self.opacities[selected_pts_mask].repeat(N,1)

        if self.transient_type == False:
            new_static_confidence = self.static_confidence[selected_pts_mask].repeat(N,1)
            self._densification_postfix({
                    "xyz": new_xyz,
                    "features_dc": new_features_dc,
                    "opacities": new_opacity,
                    "scales": new_scaling,
                    "rotations": new_rotation,
                    "static_confidence": new_static_confidence,
                    "embeddings": self.embeddings[selected_pts_mask].repeat(N,1) if self.embeddings is not None else None,
                    "features_rest": self.features_rest[selected_pts_mask].repeat(N,1) if self.features_rest is not None else None,
                })
        else:
            self._densification_postfix({
                    "xyz": new_xyz,
                    "features_dc": new_features_dc,
                    "opacities": new_opacity,
                    "scales": new_scaling,
                    "rotations": new_rotation,
                    "embeddings": self.embeddings[selected_pts_mask].repeat(N,1) if self.embeddings is not None else None,
                    "features_rest": self.features_rest[selected_pts_mask].repeat(N,1) if self.features_rest is not None else None,
                })
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * int(selected_pts_mask.sum().detach().cpu().item()), 
            device=device, dtype=torch.bool)))
        self._prune_points(prune_filter)

    def _densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        scales = self.scaling_activation(self.scales)
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        if self.config.use_gof_abs_gradient:
            assert grads_abs is not None
            assert grad_abs_threshold is not None
            selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values <= self.config.percent_dense*scene_extent)
        
        new_xyz = self.xyz[selected_pts_mask]
        new_features_dc = self.features_dc[selected_pts_mask]
        new_opacities = self.opacities[selected_pts_mask]
        new_scaling = self.scales[selected_pts_mask]
        new_rotation = self.rotations[selected_pts_mask]
        if self.transient_type == False:
            new_static_confidence = self.static_confidence[selected_pts_mask] 
            self._densification_postfix({
                "xyz": new_xyz,
                "features_dc": new_features_dc,
                "opacities": new_opacities,
                "static_confidence": new_static_confidence,
                "scales": new_scaling,
                "rotations": new_rotation,
                "embeddings": self.embeddings[selected_pts_mask] if self.embeddings is not None else None,
                "features_rest": self.features_rest[selected_pts_mask] if self.features_rest is not None else None,
            })
        else:    
            self._densification_postfix({
                "xyz": new_xyz,
                "features_dc": new_features_dc,
                "opacities": new_opacities,
                "scales": new_scaling,
                "rotations": new_rotation,
                "embeddings": self.embeddings[selected_pts_mask] if self.embeddings is not None else None,
                "features_rest": self.features_rest[selected_pts_mask] if self.features_rest is not None else None,
            })
            
    def prune_confidence(self, 
                         max_grad, 
                         min_opacity, 
                         extent, 
                         enable_size_pruning,
                         skyradius=None
                         ):
        
        del skyradius  # Unused
        grads = self.xyz_grad / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = None
        Q = None
        if self.config.use_gof_abs_gradient:
            ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
            Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        # before = self.xyz.shape[0]
        # self._densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self.xyz.shape[0]

        static_confidence = self.opacity_activation(self.static_confidence)
        scales = self.scaling_activation(self.scales)
        prune_mask = (static_confidence < min_opacity).squeeze()
        
        self._prune_points(prune_mask)

        prune = self.xyz.shape[0]
        return split - prune
        
    
    def densify_and_prune(self, 
                          max_grad, 
                          min_opacity, 
                          static_pixels,
                          extent, 
                          iteration,
                          enable_size_pruning,
                          skyradius=None):
        del skyradius  # Unused
        grads = self.xyz_grad / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = None
        Q = None
        if self.config.use_gof_abs_gradient:
            ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
            grads_abs = self.xyz_gradient_accum_abs / self.denom
            grads_abs[grads_abs.isnan()] = 0.0
            Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        before = self.xyz.shape[0]
        self._densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self.xyz.shape[0]
        self._densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self.xyz.shape[0]
        
        opacity = self.opacity_activation(self.opacities)
        scales = self.scaling_activation(self.scales)
        prune_mask = (opacity < min_opacity).squeeze()

        if enable_size_pruning:
            big_points_ws = scales.max(dim=1).values > 0.1 * extent

            # We preserve the sky points
            num_big_points = big_points_ws.sum()
            num_big_points_sky = big_points_ws.sum()
            print(f"pruning ws points {num_big_points} -> {num_big_points_sky}")

            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        self._prune_points(prune_mask)
        
        
        if iteration > 100000:
            final_confidence = compute_final_mask(static_pixels, 1 - self.static_confidence)
            prune_mask = (final_confidence < 0.02).squeeze()
            if enable_size_pruning:
                big_points_ws = scales.max(dim=1).values > 0.1 * extent

                # We preserve the sky points
                num_big_points = big_points_ws.sum()
                num_big_points_sky = big_points_ws.sum()
                print(f"pruning ws points {num_big_points} -> {num_big_points_sky}")

                prune_mask = torch.logical_or(prune_mask, big_points_ws)

            self._prune_points(prune_mask)
        
        

        prune = self.xyz.shape[0]
        return clone - before, split - clone, split - prune

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_grad[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        if self.config.use_gof_abs_gradient:
            assert self.xyz_gradient_accum_abs is not None
            assert self.xyz_gradient_accum_abs_max is not None
            self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
            self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1

    def _render_internal(self, 
                         viewpoint_camera: Cameras,
                         transient_model,
                         config: Config, 
                         *,
                         kernel_size: float, 
                         scaling_modifier = 1.0, 
                         embedding: Optional[torch.Tensor],
                         return_raw: bool = True,
                         render_depth: bool = False,
                         camera_id = None):
        
        """
        Render the scene. 
        """
        device = self.xyz.device
        assert len(viewpoint_camera.poses.shape) == 2, "Expected a single camera"
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.xyz, dtype=self.xyz.dtype, requires_grad=True, device=device) + 0
        transient_screenspace_points = torch.zeros_like(transient_model.xyz, dtype=self.xyz.dtype, requires_grad=True, device=device) + 0
        try:
            screenspace_points.retain_grad()
            transient_screenspace_points.retain_grad()
        except Exception:
            pass

        assert viewpoint_camera.image_sizes is not None, "Expected image sizes to be set"
        pose = np.copy(viewpoint_camera.poses)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
        pose = np.linalg.inv(pose)
        R = pose[:3, :3]
        T = pose[:3, 3]
        R = np.transpose(R)
        width, height = viewpoint_camera.image_sizes
        fx, fy, cx, cy = viewpoint_camera.intrinsics

        zfar = 100.0
        znear = 0.01
        trans=np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scale=1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device=device)
        projection_matrix = getProjectionMatrixFromOpenCV(width, height, fx, fy, cx, cy, znear, zfar).transpose(0, 1).to(device=device)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        # Set up rasterization configuration
        FoVx=focal2fov(float(fx), float(width))
        FoVy=focal2fov(float(fy), float(height))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        subpixel_offset = torch.zeros((int(height), int(width), 2), dtype=torch.float32, device=device)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            subpixel_offset=subpixel_offset,
            bg=torch.zeros((3,), dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=self.active_sh_degree.cpu().item(),
            campos=camera_center,
            prefiltered=False,
            debug=config.debug,
            return_accumulation=True,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        gaussians = self.get_gaussians()
        static_means3D = gaussians["xyz"]
        static_means2D = screenspace_points
        static_opacity = gaussians["opacities"]
        static_scales = gaussians["scales"]
        static_rotations = gaussians["rotations"]
        static_features = gaussians["features"].clamp_max(1.0)
        static_confidence = gaussians["static_confidence"]
        
        # fix
        transeint_gaussians = transient_model.get_gaussians()
        transient_means3D = transeint_gaussians["xyz"]
        transient_means2D = transient_screenspace_points
        transient_opacity = transeint_gaussians["opacities"]
        transient_scales = transeint_gaussians["scales"]
        transient_rotations = transeint_gaussians["rotations"]
        transient_features = transeint_gaussians["features"].clamp_max(1.0)
        
        if camera_id is not None:
            expanded_tensor = torch.tensor(camera_id).repeat(transeint_gaussians["xyz"].size()[0], 1)
            d_xyz, d_rotation, d_scaling = transient_model.deform_net(transeint_gaussians["xyz"], expanded_tensor.to(device))
            
            transient_means3D = transeint_gaussians["xyz"] + d_xyz
            transient_scales = transeint_gaussians["scales"] + d_scaling
            transient_rotations = transeint_gaussians["rotations"] + d_rotation
        
        # Concatenation
        static_dir_pp_normalized = F.normalize(static_means3D - camera_center.repeat(static_features.shape[0], 1), dim=1)
        transient_dir_pp_normalized = F.normalize(transient_means3D - camera_center.repeat(transient_features.shape[0], 1), dim=1)   
        
        # fix
        if static_features.shape[-1] == 3:
            static_all_colors = static_features.clamp_max(1.0)
        else:
            shdim = (self.config.sh_degree + 1) ** 2
            assert static_features.shape[-1] == shdim * 3
            shs_view = static_features.view(-1, shdim, 3).transpose(1, 2).contiguous()
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, static_dir_pp_normalized)
            static_all_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
            transient_embedding_expanded = assert_not_none(embedding)[None].repeat(len(transient_means2D), 1)
            transient_colors_toned = self.appearance_mlp(transient_model.embeddings, transient_embedding_expanded, transient_features).clamp_max(1.0)
            shs_view = transient_colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous()
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, transient_dir_pp_normalized)
            transient_all_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    
        if self.dataset_type == "nerfonthego":
            # fix Static, Transient
            static_raw_rendered_image, static_radii, static_accumulation, static_pixels = rasterizer(
                means3D=static_means3D,
                means2D=static_means2D,
                shs=None,
                colors_precomp=static_all_colors,
                opacities=static_opacity,
                scales=static_scales,
                rotations=static_rotations,
                cov3D_precomp=None)
            static_rendered_image = static_raw_rendered_image 
            
            
            # confidnce map..
            _, _, confidence_rendered_image, pixels = rasterizer(
                means3D=static_means3D,
                means2D=static_means2D,
                shs=None,
                colors_precomp=static_all_colors,
                opacities=static_confidence,
                scales=static_scales,
                rotations=static_rotations,
                cov3D_precomp=None)
            
            transient_raw_rendered_image, transient_radii, transient_accumulation, pixels = rasterizer(
                means3D=transient_means3D,
                means2D=transient_means2D,
                shs=None,
                colors_precomp=transient_all_colors,
                opacities=transient_opacity,
                scales=transient_scales,
                rotations=transient_rotations,
                cov3D_precomp=None)
            transient_rendered_image = transient_raw_rendered_image 
            
        if self.dataset_type == "nerfonthego": pass
        else:            
            if self.config.appearance_enabled:
                
                static_embedding_expanded = assert_not_none(embedding)[None].repeat(len(static_means2D), 1)       # 전체
                transient_embedding_expanded = assert_not_none(embedding)[None].repeat(len(transient_means2D), 1)       # 전체
                
                
                # Static
                static_colors_toned = self.appearance_mlp(self.embeddings, static_embedding_expanded, static_features).clamp_max(1.0)
                shdim = (self.config.sh_degree + 1) ** 2
                static_colors_toned = static_colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
                static_colors_toned = eval_sh(self.active_sh_degree, static_colors_toned, static_dir_pp_normalized)
                static_colors_toned = torch.clamp_min(static_colors_toned + 0.5, 0.0)

                # Transient
                transient_colors_toned = self.appearance_mlp(transient_model.embeddings, transient_embedding_expanded, transient_features).clamp_max(1.0)
                shdim = (self.config.sh_degree + 1) ** 2
                transient_colors_toned = transient_colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
                transient_colors_toned = eval_sh(self.active_sh_degree, transient_colors_toned, transient_dir_pp_normalized)
                transient_colors_toned = torch.clamp_min(transient_colors_toned + 0.5, 0.0)
                static_rendered_image, static_radii, static_accumulation, static_pixels = rasterizer(
                    means3D = static_means3D,
                    means2D = static_means2D,
                    colors_precomp = static_colors_toned,
                    opacities = static_opacity,
                    scales = static_scales,
                    rotations = static_rotations,
                    shs = None,
                    cov3D_precomp = None)
                
                _, _, confidence_rendered_image, _ = rasterizer(
                    means3D = static_means3D,
                    means2D = static_means2D,
                    colors_precomp = static_colors_toned,
                    opacities = static_confidence,
                    scales = static_scales,
                    rotations = static_rotations,
                    shs = None,
                    cov3D_precomp = None)
                
                transient_rendered_image, transient_radii, transient_accumulation, transient_pixels = rasterizer(
                    means3D = transient_means3D,
                    means2D = transient_means2D,
                    colors_precomp = transient_colors_toned,
                    opacities = transient_opacity,
                    scales = transient_scales,
                    rotations = transient_rotations,
                    shs = None,
                    cov3D_precomp = None)
                
                

        static_visibility_filter = assert_not_none(static_radii) > 0
        transient_visibility_filter = assert_not_none(transient_radii) > 0
        
        out = {"render": static_rendered_image,
               "static_pixel": static_pixels,
               "static_visibility_filter": static_visibility_filter,
               "transient_visibility_filter": transient_visibility_filter,
               "static_viewspace_points": static_means2D, 
               "transient_viewspace_points": transient_means2D, 
               "static_accumulation": static_accumulation,
               "transient_accumulation": transient_accumulation,
               "static_radii": assert_not_none(static_radii),
               "transient_radii": assert_not_none(transient_radii),
               "static_render": static_rendered_image,
               "transient_render": transient_rendered_image,
               "confidence_rendered_image": confidence_rendered_image
               }
        if return_raw:
            out["raw_render"] = static_rendered_image

        if render_depth:
            dist = torch.norm(static_means3D - camera_center[None], dim=-1).unsqueeze(-1).repeat(1, 3)
            out["depth"] = rasterizer(
                means3D=static_means3D,
                means2D=static_means2D,
                colors_precomp=dist,
                opacities=static_opacity,
                scales=static_scales,
                rotations=static_rotations,
                shs=None,
                cov3D_precomp=None)[0][0]
        return out


class ForestSplats(Method):
    config_overrides: Optional[dict] = None

    def __init__(self,
                 *,
                 checkpoint: Optional[Path] = None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None,
                 dataset_type,
                 save_path,
                 total_iteration):
        device = torch.device("cuda")

        self.dataset_type = dataset_type
        self.to_pil = ToPILImage()
        self.optimizer = None
        self.checkpoint = checkpoint
        self.save_path = save_path
        self.total_iteration = total_iteration
        self.step = 0
        self.bce_loss = nn.BCELoss()
        self.Wongi_loss = WongiLoss()
        
        # Setup parameters
        load_state_dict = None
        self.config: Config = OmegaConf.structured(Config)
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            logging.info(f"Loading config file {os.path.join(checkpoint, 'config.yaml')}")
            self.config = cast(Config, OmegaConf.merge(self.config, OmegaConf.load(os.path.join(checkpoint, "config.yaml"))))
            self._loaded_step = self.step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(str(self.checkpoint)) if x.startswith("chkpnt-"))[-1]
            state_dict_name = f"chkpnt-{self._loaded_step}.pth"
            load_state_dict = torch.load(os.path.join(checkpoint, state_dict_name))
        else:
            if config_overrides is not None:
                if "config" in config_overrides:
                    config_overrides = dict(config_overrides)
                    config_file = config_overrides.pop("config")
                else:
                    config_file = "default.yml"
                logging.info(f"Loading config file {config_file}")
                config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", config_file)
                self.config = cast(Config, OmegaConf.merge(self.config, OmegaConf.load(config_file)))
                oc_config_overrides = OmegaConf.from_dotlist([f"{k}={v}" for k, v in config_overrides.items()])
                self.config = cast(Config, OmegaConf.merge(self.config, oc_config_overrides))

        self._viewpoint_stack = []

        self.train_cameras = None
        self.cameras_extent = None

        # Used for saving
        self._json_cameras = None

        # Initialize system state (RNG)
        safe_state()

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        self.model = GaussianModel(self.config, dataset_type, train_dataset is not None).to(device)
        self.transient_model = GaussianModel(self.config, dataset_type, train_dataset is not None, transient_type = True).to(device)       # fix
        self._sky_distance = None
        # self.appearance_module = E_attr(input_dim_a = 3).to(device)
    
        if train_dataset is not None:
            self._setup_train(train_dataset, load_state_dict)
        elif load_state_dict is not None:
            self.model.load_state_dict(load_state_dict)
            
        with torch.no_grad():
            self.vgg11 = models.vgg11(pretrained=True).to(device)
            self.vgg11.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),  
                nn.ReLU(True),
                nn.Linear(4096, 32),         
            ).to(device)
            
    def _setup_train(self, train_dataset: Dataset, load_state_dict):
        points3D_xyz = train_dataset["points3D_xyz"]
        points3D_rgb = train_dataset["points3D_rgb"]
        assert points3D_xyz is not None, "Train points3D_xyz are required for training"
        assert points3D_rgb is not None, "Train points3D_rgb are required for training"
        opacities = 0.1 * np.ones(len(points3D_xyz))
        static_confidence = 0.1 * np.ones(len(points3D_xyz))
        if self.config.num_sky_gaussians:
            th_cameras = train_dataset["cameras"].apply(lambda x, _: torch.from_numpy(x).cuda())
            skybox, self._sky_distance = get_sky_points(self.config.num_sky_gaussians, torch.from_numpy(points3D_xyz).cuda(), th_cameras)
            skybox = skybox.cpu().numpy()
            skycolor = np.array([[237,247,252]], dtype=np.uint8).repeat(skybox.shape[0], axis=0)
            logging.info(f"Adding skybox with {skybox.shape[0]} points")
            train_dataset = train_dataset.copy()
            train_dataset["points3D_xyz"] = np.concatenate((points3D_xyz, skybox))
            train_dataset["points3D_rgb"] = np.concatenate((points3D_rgb, skycolor))
            opacities = np.concatenate((opacities, 1.0 * np.ones(skybox.shape[0])))
            static_confidence = np.concatenate((static_confidence, 1.0 * np.ones(skybox.shape[0])))

        # import trimesh
        # _ = trimesh.PointCloud(skybox).export("/tmp/back.ply")
        # breakpoint()
        # Prepare dataset
        self.cameras_extent = get_cameras_extent(train_dataset["cameras"])
        self.train_cameras = train_dataset["cameras"]
        self.train_images = [
            torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)) for img in train_dataset["images"]
        ]
        self.train_sampling_masks = None
        if train_dataset["sampling_masks"] is not None:
            self.train_sampling_masks = [
                torch.from_numpy(convert_image_dtype(img, np.float32)[None]) for img in train_dataset["sampling_masks"]
            ]
        # Clear memory
        train_dataset["images"] = None  # type: ignore
        train_dataset["sampling_masks"] = None  # type: ignore

        # Setup model
        if self.checkpoint is None:
            xyz, colors = train_dataset["points3D_xyz"], train_dataset["points3D_rgb"]
            self.model.initialize_from_points3D(xyz, colors, self.cameras_extent, opacities=opacities, static_confidence = static_confidence)
            self.model.set_num_training_images(len(train_dataset["cameras"]))
            
            # fix
            num_samples = 5000
            indices = np.random.choice(xyz.shape[0], num_samples, replace=False)  # 중복 없이 샘플링
            self.transient_model.initialize_from_points3D(xyz[indices], colors[indices], self.cameras_extent, opacities=opacities[indices])
            self.transient_model.set_num_training_images(len(train_dataset["cameras"]))
        else:
            self.model.load_state_dict(load_state_dict)

        self._viewpoint_stack = []

        self.model.compute_3D_filter(cameras=self.train_cameras)
        # fix
        self.transient_model.compute_3D_filter(cameras=self.train_cameras)

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        return MethodInfo(
            method_id="wild-gaussians",  # Will be filled by the registry
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            **self.get_method_info(),
            num_iterations=self.config.iterations,
            loaded_step=self._loaded_step,
        )

    def optimize_embedding(
        self, 
        dataset: Dataset, *,
        embedding: Optional[np.ndarray] = None
    ) -> OptimizeEmbeddingOutput:
        device = self.model.xyz.device
        camera = dataset["cameras"].item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"

        self.model.eval()
        self.transient_model.eval()
        i = 0
        losses, psnrs, mses = [], [], []
        appearance_embedding = (
            torch.from_numpy(embedding).to(device) if embedding is not None else self.model.get_embedding(None)
        )
        
        if self.config.appearance_enabled:
            appearance_embedding_param = torch.nn.Parameter(assert_not_none(appearance_embedding).requires_grad_(True))
            optimizer = torch.optim.Adam([appearance_embedding_param], lr=self.config.appearance_embedding_optim_lr)
            
            gt_image = torch.tensor(convert_image_dtype(dataset["images"][i], np.float32), dtype=torch.float32, device=device).permute(2, 0, 1)
            gt_mask = torch.tensor(convert_image_dtype(dataset["sampling_masks"][i], np.float32), dtype=torch.float32, device=device)[..., None].permute(2, 0, 1) if dataset["sampling_masks"] is not None else None

            with torch.enable_grad():
                app_optim_type = self.config.appearance_optim_type
                loss_mult = None
                if app_optim_type.endswith("-scaled"):
                    app_optim_type = app_optim_type[:-7]
                    if self.model.uncertainty_model is not None:
                        _, _, loss_mult = self.model.uncertainty_model.get_loss(gt_image, gt_image)
                        loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype)
                for _ in range(self.config.appearance_embedding_optim_iters):
                    optimizer.zero_grad()     # fix
                    image = self.model._render_internal(camera, self.transient_model, config=self.config, embedding=appearance_embedding_param, 
                                                        kernel_size=self.config.kernel_size)["render"]
                    # viewpoint_cam, self.transient_model, config=self.config, embedding = embedding_intrinsic, kernel_size=self.config.kernel_size, transient_embedding = transient_embedding, appearance_feature = appearance_feature)
                    if gt_mask is not None:
                        image = scale_grads(image, gt_mask.float())
                    if loss_mult is not None:
                        image = scale_grads(image, loss_mult)

                    mse = torch.nn.functional.mse_loss(image, gt_image)


                    if app_optim_type == "mse":
                        loss = mse
                    elif app_optim_type == "dssim+l1":
                        Ll1 = torch.nn.functional.l1_loss(image, gt_image)
                        ssim_value = ssim(image, gt_image, size_average=True)
                        loss = (
                            (1.0 - self.config.lambda_dssim) * Ll1 +
                            self.config.lambda_dssim * (1.0 - ssim_value)
                        )
                    else:
                        raise ValueError(f"Unknown appearance optimization type {app_optim_type}")
                    loss.backward()
                    # TODO: use uncertainty here as well
                    # print(float(appearance_embedding_param.grad.abs().max().cpu()), float(mse.cpu()))
                    optimizer.step()

                    losses.append(loss.detach().cpu().item())
                    mses.append(mse.detach().cpu().item())
                    psnrs.append(20 * math.log10(1.0) - 10 * torch.log10(mse).detach().cpu().item())

            if self.model.optimizer is not None:
                self.model.optimizer.zero_grad()
            appearance_embedding = appearance_embedding_param
            embedding_np = appearance_embedding_param.detach().cpu().numpy()
            return {
                "embedding": embedding_np,
                "metrics": {
                    "psnr": psnrs,
                    "mse": mses,
                    "loss": losses,
                }
            }
        else:
            raise NotImplementedError("Trying to optimize embedding with appearance_enabled=False")

    def render(self, camera: Cameras, options=None, **kwargs) -> RenderOutput:
        del kwargs
        camera = camera.item()
        device = self.model.xyz.device
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        render_depth = False
        if options is not None and "depth" in options.get("outputs", ()):
            render_depth = True

        self.model.eval()
        self.transient_model.eval()
        with torch.no_grad():
            _np_embedding = (options or {}).get("embedding", None)
            embedding = (
                torch.from_numpy(_np_embedding)
                if _np_embedding is not None else self.model.get_embedding(None)
            )
            del _np_embedding
            
            embedding = embedding.to(device) if embedding is not None else None
            
            if self.dataset_type != "nerfonthego":
                out = self.model._render_internal(camera, 
                                              transient_model= self.transient_model,
                                              config=self.config, 
                                              embedding=embedding, 
                                              kernel_size=self.config.kernel_size, 
                                              render_depth=render_depth,)
        #         image = self.model._render_internal(camera, self.transient_model, config=self.config, embedding=embedding_intrinsic, 
        # kernel_size=self.config.kernel_size, transient_embedding = transient_embedding, appearance_feature = appearance_feature)["render"]
            else:
                out = self.model._render_internal(camera, 
                                              transient_model= self.transient_model,
                                              config=self.config, 
                                              embedding=embedding, 
                                              kernel_size=self.config.kernel_size, 
                                              render_depth=render_depth)
                
            # out = self.model._render_internal(camera, 
            #                                   transient_model= self.transient_model,
            #                                   config=self.config, 
            #                                   embedding=embedding, 
            #                                   kernel_size=self.config.kernel_size, 
            #                                   render_depth=render_depth)
            image = out["render"]
            image = torch.clamp(image, 0.0, 1.0).nan_to_num_(0.0)

            color = image.detach().permute(1, 2, 0).cpu().numpy()

            ret_out: RenderOutput = {
                "color": color,
                "accumulation": out["static_accumulation"].squeeze(-1).detach().cpu().numpy(),
            }
            if out.get("depth") is not None:
                ret_out["depth"] = out["depth"].detach().cpu().numpy()
            return ret_out

    def _get_viewpoint_stack(self, step: int):
        assert self.train_cameras is not None, "Method not initialized"
        generator = torch.Generator()
        generator.manual_seed(step // 300)
        num_images = 30
        indices = torch.multinomial(
            torch.ones(len(self.train_cameras), dtype=torch.float32), 
            num_images,
            generator=generator,
        )
        return [self.train_cameras[i] for i in indices]

    def train_iteration(self, step):
        assert self.train_cameras is not None, "Method not initialized"
        assert self.model.optimizer is not None, "Method not initialized"
        assert self.train_cameras.image_sizes is not None, "Image sizes are required for training"
 
        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step
        self.model.train()

        self.model.update_learning_rate(iteration)
        device = self.model.xyz.device
        # fix
        self.transient_model.train()
        self.transient_model.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.model.oneupSHdegree()
            self.transient_model.oneupSHdegree()
 
        # Pick a random Camera
        if not self._viewpoint_stack:
            self._viewpoint_stack = list(range(len(self.train_cameras)))
            
        camera_id = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))
        viewpoint_cam = self.train_cameras[camera_id]
        assert viewpoint_cam.image_sizes is not None, "Image sizes are required for training"
        embedding_intrinsic = self.model.get_embedding(self.dataset_type, train_image_id=camera_id)
        fid = int(camera_id) / (len(self.train_cameras) - 1)
        render_pkg = self.model._render_internal(viewpoint_cam, self.transient_model, 
                                                 config=self.config, embedding = embedding_intrinsic, 
                                                 kernel_size=self.config.kernel_size, render_depth = True, camera_id = fid)
        
        accumulation_img = render_pkg["static_accumulation"]
        confidence_rendered_image = render_pkg["confidence_rendered_image"]
        static_image_toned, transient_image_toned = render_pkg["static_render"], render_pkg["transient_render"],
        static_viewspace_points, transient_viewspace_points = render_pkg["static_viewspace_points"], render_pkg["transient_viewspace_points"],   
        static_visibility_filter, transient_visibility_filter = render_pkg["static_visibility_filter"], render_pkg["transient_visibility_filter"],   
        static_radii, transient_radii = render_pkg["static_radii"], render_pkg["transient_radii"]
        static_depth = render_pkg["depth"]
        static_pixels = render_pkg["static_pixel"]
        
        # Loss
        gt_image = self.train_images[camera_id].to(device)
        sampling_mask = self.train_sampling_masks[camera_id].to(device) if self.train_sampling_masks is not None else None 

        # Apply mask
        if sampling_mask is not None:
            image = scale_grads(image, sampling_mask)
            image_toned = scale_grads(image_toned, sampling_mask)

        uncertainty_loss = 0
        metrics = {}
        loss_mult: Any = 1.
        if self.model.uncertainty_model is not None:
            del loss_mult
            uncertainty_loss, metrics, loss_mult = self.model.uncertainty_model.get_loss(gt_image, image_toned.detach(), iteration, _cache_entry=('train', camera_id))
            loss_mult = (loss_mult > 1).to(dtype=loss_mult.dtype)

            if iteration < self.config.uncertainty_warmup_start:
                loss_mult = 1
            elif iteration < self.config.uncertainty_warmup_start + self.config.uncertainty_warmup_iters:
                p = (iteration - self.config.uncertainty_warmup_start) / self.config.uncertainty_warmup_iters
                loss_mult = 1 + p * (loss_mult - 1)
            if self.config.uncertainty_center_mult:
                loss_mult = loss_mult.sub(loss_mult.mean() - 1).clamp(0, 2)
            if self.config.uncertainty_scale_grad:
                image = scale_grads(image, loss_mult)
                image_toned = scale_grads(image_toned, loss_mult)
                loss_mult = 1

        cof_mask = self.transient_model.implict_mask(gt_image.unsqueeze(0)).squeeze(0).squeeze(0).to(device)                # Feature Masking
        
        # Masking Mask
        if iteration > 1:
            pixel_mask = cof_mask.detach().cpu().clone()
            filed_mask = generate_filled_mask(gt_image.cpu(), pixel_mask,  num_segments= 30)
            combined_image = filed_mask * transient_image_toned + (1 - filed_mask) * static_image_toned
        else:
            combined_image = cof_mask * transient_image_toned + (1 - cof_mask) * static_image_toned
            
        last_densify_iter = min(iteration, self.config.densify_until_iter - 1)
        last_dentify_iter = (last_densify_iter // self.config.opacity_reset_interval) * self.config.opacity_reset_interval
        if iteration < last_dentify_iter + self.config.uncertainty_protected_iters:
            # Keep track of max radii in image-space for pruning
            try:
                uncertainty_loss = uncertainty_loss.detach()  # type: ignore
            except AttributeError:
                pass
   
        
        static_Ll1 = torch.nn.functional.l1_loss((1 - cof_mask) * static_image_toned, (1 - cof_mask) * gt_image, reduction='none')
        static_ssim_value = ssim((1 - cof_mask) * static_image_toned,  (1 - cof_mask) * gt_image, size_average=False) 
        static_loss = ((1.0 - self.config.lambda_dssim) * (static_Ll1 * loss_mult).mean() + 
                        self.config.lambda_dssim * ((1.0 - static_ssim_value) * loss_mult).mean())
        
        transient_Ll1 = torch.nn.functional.l1_loss(cof_mask * transient_image_toned, cof_mask * gt_image, reduction='none')
        
        loss, _ = self.Wongi_loss(combined_image, gt_image, cof_mask, iteration)
        loss += (0.2 * static_loss)
        loss += (transient_Ll1.mean())
        
            
        loss.backward()
        with torch.no_grad():
            mse = (combined_image - gt_image).pow_(2)
            
            psnr_value = 20 * math.log10(1.) - 10 * torch.log10(mse.mean())
            metrics.update({
                "l1_loss": static_Ll1.detach().mean().cpu().item(), 
                "ssim": static_ssim_value.detach().mean().cpu().item(),
                "mse": mse.detach().mean().cpu().item(),
                "loss": loss.detach().cpu().item(), 
                "psnr": psnr_value.detach().cpu().item(),
                "num_gaussians": len(self.model.xyz),
            })

            def _reduce_masked(tensor, mask):
                return (
                    (tensor * mask).sum() / mask.sum()
                ).detach().cpu().item()

            if sampling_mask is not None:
                mask_percentage = sampling_mask.detach().mean().cpu().item()
                metrics["mask_percentage"] = mask_percentage
                metrics["ssim_masked"] = _reduce_masked(ssim_value, sampling_mask)
                metrics["mse_masked"] = masked_mse = _reduce_masked(mse, sampling_mask)
                masked_psnr_value = 20 * math.log10(1.) - 10 * math.log10(masked_mse)
                metrics["psnr_masked"] = masked_psnr_value
                metrics["l1_loss_masked"] = _reduce_masked(Ll1, sampling_mask)

            # Densification
            if iteration < self.config.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.model.max_radii2D[static_visibility_filter] = torch.max(self.model.max_radii2D[static_visibility_filter], static_radii[static_visibility_filter])
                self.model.add_densification_stats(static_viewspace_points, static_visibility_filter)
                                

                if iteration > self.config.densify_from_iter and iteration % self.config.densification_interval == 0:
                    self.model.densify_and_prune(
                        self.config.densify_grad_threshold,  
                        0.005,
                        static_pixels,
                        self.cameras_extent,
                        iteration,
                        enable_size_pruning=iteration > self.config.opacity_reset_interval,
                        skyradius=self._sky_distance)
                    self.model.compute_3D_filter(cameras=self.train_cameras)

                    

                if iteration % self.config.opacity_reset_interval == 0:
                    self.model.reset_opacity()
                    # fix
                    # self.transient_model.reset_opacity()

            if iteration % 100 == 0 and iteration > self.config.densify_until_iter:
                if iteration < self.config.iterations - 100:
                    # don't update in the end of training
                    self.model.compute_3D_filter(cameras=self.train_cameras)
                    # self.transient_model.compute_3D_filter(cameras=self.train_cameras)

            # Optimizer step
            if iteration < self.config.iterations:
                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)
                
                self.transient_model.optimizer.step()
                self.transient_model.optimizer.zero_grad(set_to_none=True)

        self.step = self.step + 1
        self.model.eval()
        self.transient_model.eval()      # Eval
        return metrics

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        embed = self.model.get_embedding(index)
        if embed is not None:
            return embed.detach().cpu().numpy()
        return embed
    

    def save(self, path):
        self.model.save_ply(os.path.join(path, "point_cloud.ply"))
        self.transient_model.save_ply(os.path.join(path, "transient_point.ply"))
        ckpt = self.model.state_dict()
        ckpt_path = str(path) + f"/chkpnt-{self.step}.pth"
        torch.save(ckpt, ckpt_path)
        OmegaConf.save(self.config, os.path.join(path, "config.yaml"))

        # Note, since the torch checkpoint does not have deterministic SHA, we compute the SHA here.
        sha = get_torch_checkpoint_sha(ckpt)
        with open(ckpt_path + ".sha256", "w", encoding="utf8") as f:
            f.write(sha)


def tensor_to_uncertainty_img(tensor, iteration, camera_id, loss_name):
    # 텐서를 numpy array로 변환하고, (C, H, W) -> (H, W, C) 형식으로 변경
    image_np = tensor.permute(1, 2, 0).detach().cpu().numpy()  # shape: (262, 172, 3)
    image_np = (image_np * 255).astype(np.uint8)
    # image_np = image_tensor.cpu().numpy()  # numpy array로 변환
    # image_np = np.uint8(image_np * 255)  # [0,1] 범위인 경우 [0,255] 범위로 변환
    if not os.path.exists(f"./wang/uncertainty/{str(loss_name)}"):
        os.makedirs(f"./wang/uncertainty/{str(loss_name)}")
    else:
        pass
    cv2.imwrite(f"./wang/uncertainty/{str(loss_name)}/image_{str(iteration)}_{str(camera_id)}.png", image_np)    
    
def tensor_to_gt_rgb(save_path, tensor, iteration, camera_id):
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image_np = np.array(image)
    
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if not os.path.exists(f"./{save_path}/gt"):
        os.makedirs(f"./{save_path}/gt")
        cv2.imwrite(f"./{save_path}/gt/gt_image_{iteration}_{camera_id}.png", image_np)
    else:
        cv2.imwrite(f"./{save_path}/gt/gt_image_{iteration}_{camera_id}.png", image_np)
        
        
def save_mask(save_path, tensor, iteration, name):
    transform = transforms.ToPILImage()
    image = transform(tensor)
    image_np = np.array(image)
    
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if not os.path.exists(f"./{save_path}/{name}"):
        os.makedirs(f"./{save_path}/{name}")
    else:
        pass
    cv2.imwrite(f"./{save_path}/{name}/masking_{iteration}.png", image_np)


def make_mask(image, gt_image, transient_image, device):
    _msssim = msssim(gt_image.unsqueeze(0), image.unsqueeze(0), max_size=400, min_size=80).unsqueeze(1).squeeze(0).permute(1, 2, 0).squeeze(0).cpu().detach().numpy()
    _msssim = (_msssim * 255).astype(np.uint8)      # blurred_image = cv2.GaussianBlur(_msssim, (9, 9), 5)
    _, car_otsu_thresh = cv2.threshold(_msssim, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _msssim = ~(_msssim.squeeze(-1))
    threshold_image = Image.fromarray(~car_otsu_thresh)
    car_otsu_thresh = np.expand_dims(car_otsu_thresh, axis=0) 
    car_otsu_thresh = torch.tensor(car_otsu_thresh, dtype=torch.float32) / 255.0
    car_otsu_thresh_3d = car_otsu_thresh.repeat(3, 1, 1).to(device)
    masked_gt_image = gt_image * car_otsu_thresh_3d
    masked_transient_image_toned = transient_image * car_otsu_thresh_3d
    
    return masked_gt_image, masked_transient_image_toned, np.array(threshold_image), _msssim

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def compute_final_mask(static_pixels, static_confidence):
    pixel_weights = torch.nn.functional.softmax(static_pixels, dim=0)
    weighted_mask = pixel_weights * static_confidence
    final_mask = weighted_mask / weighted_mask.sum()
    return final_mask        


def generate_filled_mask(image: np.ndarray, mask: np.ndarray, num_segments=20) -> np.ndarray:
    # SLIC Superpixel Segmentation
    segments = slic(image, n_segments=num_segments, compactness=1, sigma=3, start_label=0)
    filled_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for region in regionprops(segments):
        coords = region.coords  
        
        coords[:, 0] = np.clip(coords[:, 0], 0, mask.shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, mask.shape[1] - 1)
        mask_pixels = mask[coords[:, 0], coords[:, 1]]

        if isinstance(mask_pixels, torch.Tensor):
            mask_pixels = mask_pixels.cpu().numpy()
        mask_ratio = np.sum(mask_pixels) / len(mask_pixels)
        
        if mask_ratio >= 0.2:
            region_mask = np.zeros_like(mask, dtype=np.float32)
            region_mask[coords[:, 0], coords[:, 1]] = mask[coords[:, 0], coords[:, 1]]

            region_mask_torch = torch.tensor(region_mask).unsqueeze(0).unsqueeze(0)
            h, w = mask.shape
            grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            grid = np.stack((grid_x, grid_y), axis=-1)
            grid_torch = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)  # (1, h, w, 2)

            interpolated_mask = F.grid_sample(region_mask_torch, grid_torch, mode='bilinear', align_corners=False)
            interpolated_mask_np = interpolated_mask.squeeze().numpy()

            filled_mask[coords[:, 0], coords[:, 1]] = np.max(interpolated_mask_np[coords[:, 0], coords[:, 1]]).astype(np.uint8)
        else:
            filled_mask[coords[:, 0], coords[:, 1]] = mask[coords[:, 0], coords[:, 1]]
    
    return filled_mask
