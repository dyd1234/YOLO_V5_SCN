# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

import inspect
from torch import Tensor

from torch.cuda.amp import autocast
from typing import List
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import ( # 
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
    C3_SCN, # 4 new structures
    SPPF_SCN,
    Conv_SCN,
    Bottleneck_SCN,
    Conv_SCN_B, # conv_2d with bias
    # SCN,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import ( # 
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync, # 
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module): #
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode 

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True): # 
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors 
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid 
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x): # 
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    # print(f"Before operation - xy device: {xy.device}")
                    # print(f"self.grid[{i}] device: {self.grid[i].device}")
                    # print(f"self.stride[{i}] device: {self.stride[i].device}")
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # print(f"After operation - xy device: {xy.device}")
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    """YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):

            def _forward(x):
                """Passes the input 'x' through the model and returns the processed output."""
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False): 
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y): 
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None): 
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


# -------------------------------------------------------------------------------------
# SCN part 
# define SCN direct inside the head (detection) # 
class Detect_SCN(nn.Module): # do need to have forward and scn_forward 
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, dimension, nc=80, anchors=(), ch=(), inplace=True, is_scn=False): # if we SCN all the parts
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.requires_hyper_x = True # used by the upper layer calling 
        self.nc = nc  # number of classes
        # print(f"The nc and its type is {nc} and {type(nc)}")
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output 
        # self.m = nn.ModuleList(Conv_SCN_B(dimension, x, self.no * self.na, 1, is_scn=is_scn) for x in ch)  # output 
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        # set some other stuff for the name of the 
        self.kernel_size = 1

        # define the weights list and bias list
        x1, x2, x3 = ch #
        # # first detection 
        # self.conv_weight_list1 = self.create_param_combination_conv2d(dimension, x1, self.no * self.na, 1)
        # self.bias_list1 = self.create_param_combination_bias(dimension, self.no * self.na)

        # self.conv_weight_list2 = self.create_param_combination_conv2d(dimension, x2, self.no * self.na, 1)
        # self.bias_list2 = self.create_param_combination_bias(dimension, self.no * self.na)

        # self.conv_weight_list3 = self.create_param_combination_conv2d(dimension, x3, self.no * self.na, 1)
        # self.bias_list3 = self.create_param_combination_bias(dimension, self.no * self.na)

        # self.conv_weight_list = [] # 
        # self.conv_bias_list = [] # 
        self.conv_weight_list = nn.ParameterList() # OK
        self.conv_bias_list = nn.ParameterList() # 

        # 
        for x in ch: # 
            self.conv_weight_list.append(self.create_param_combination_conv2d(dimension, x, self.no * self.na, 1))
            self.conv_bias_list.append(self.create_param_combination_bias(dimension, self.no * self.na))

        # self._initialize_biases() # might be OK not very good

    def forward(self, x, hyper_x): # 
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            # x[i] = self.m[i](x[i], hyper_x)  # conv
            x[i] = self.execute_hyper_conv2d(x[i], # 
                                      self.conv_weight_list[i],
                                      self.conv_bias_list[i],
                                      hyper_x,
                                      )

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)

                    # print(f"Before operation - xy device: {xy.device}")
                    # print(f"self.grid[{i}] device: {self.grid[i].device}")
                    # print(f"self.stride[{i}] device: {self.stride[i].device}")
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    # print(f"After operation - xy device: {xy.device}")
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape 
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
    # try tomorrow morning 
    # def _initialize_biases(self, cf=None):
    #     """
    #     Initializes biases, optionally using class frequencies (cf).
    #     This logic mirrors the YOLOv5 Detect module's bias initialization.
    #     """
    #     for bias_list in self.conv_bias_list:  # if this is OK
    #         for bias in bias_list:
    #             b = bias.data.view(self.na, -1)  # 确保 bias 是二维张量
    #             b[:, 4] += math.log(8 / (640 / self.stride) ** 2)  # obj (8 objects per 640 image)
    #             b[:, 5:5 + self.nc] += (
    #                 math.log(0.6 / (self.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
    #             )  # cls
    #             bias.data = b.view(-1)

    # # the additional things for conv2d with 
    # def _initialize_biases(self, cf=None): # 
    #     """
    #     Initializes biases, optionally using class frequencies (cf).
    #     This logic mirrors the YOLOv5 Detect module's bias initialization. 
    #     """
    #     for bias_list in self.conv_bias_list: # if this is OK
    #         for bias in bias_list:
    #             bias.data.view(-1)[:, 4] += math.log(8 / (640 / self.stride) ** 2)  # obj (8 objects per 640 image)
    #             bias.data.view(-1)[:, 5 : 5 + self.c2] += (
    #                 math.log(0.6 / (self.c2 - 0.99999)) if cf is None else torch.log(cf / cf.sum())
    #             )  # cls

    def create_param_combination_conv2d(self, dimensions: int, in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.ParameterList:
        weight_list = nn.ParameterList()
        for _ in range(dimensions):
            weight = Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
            init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)
        return weight_list
    
    def create_param_combination_bias(self, dimensions: int, out_channels: int) -> nn.ParameterList:
        bias_list = nn.ParameterList()
        for _ in range(dimensions):
            bias = Parameter(torch.empty(out_channels))
            fan_in = self.kernel_size * self.kernel_size * out_channels
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return bias_list
    
    # The list is not OK
    def calculate_weighted_sum(self, param_list: List[Parameter], coefficients: torch.Tensor) -> torch.Tensor:
        with autocast():
            weighted_list = [a * b for a, b in zip(param_list, coefficients)]
            return torch.sum(torch.stack(weighted_list), dim=0)

    def execute_hyper_conv2d(self, x: torch.Tensor, weight_list: List[Parameter], bias_list: List[Parameter], coefficients: torch.Tensor, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> torch.Tensor:
        # print(f"x device: {x.device}, weight_list device: {weight_list[0].device}, bias_list device: {bias_list[0].device}, coefficients device: {coefficients.device}")  # 调试信息
        weights = self.calculate_weighted_sum(weight_list, coefficients)
        bias = self.calculate_weighted_sum(bias_list, coefficients)

        with autocast():
            return F.conv2d(x, weight=weights, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# YOLO v5 base model
# This part is not very clear, still has some problems
class BaseModel_SCN(nn.Module): # 重新设计一下如何区分scn和非scn的模型
    """YOLOv5 base model."""

    def forward(self, x, hyper_x=None, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        # return self._forward_once(x, profile, visualize)  # single-scale inference, train
        # I think no need to have diferent 
        if hyper_x is not None: # t
            return self._forward_once_SCN(x, hyper_x, profile, visualize)  # for SCN mode
        else:
            return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # the will be reused in the detection_model
    # most inportant part of the forward pass
    def _forward_once(self, x, hyper_x=None, profile=False, visualize=False): # 
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            
            # x = m(x)  # run
            # if hyper_x is not None and hasattr(m, 'requires_hyper_x') and m.requires_hyper_x: # 
            # if m.__class__.__name__ in isinstance(Conv_SCN, C3_SCN, SPPF_SCN, Bottleneck_SCN, Conv_SCN_B): # need hyper_x
            if isinstance(m, (Conv_SCN, C3_SCN, SPPF_SCN, Bottleneck_SCN, Conv_SCN_B, Detect_SCN)):
                # print(f"ccThe requires_hyper_x is {m.requires_hyper_x}")
                # print(f"ccThe requires_hyper_x is {hasattr(m, 'requires_hyper_x')}")
                # print(f"ccThe requires_hyper_x is {hyper_x is not None}")
                # print(f"ccThe layer is {m} --------------------------------------------------------------")
                x = m(x, hyper_x) # check here for the name is OK
            else:
                # print(f"The layer is {m}")
                # print(f"The requires_hyper_x is {m.requires_hyper_x}")
                # print(f"The requires_hyper_x is {hasattr(m, 'requires_hyper_x')}")
                # print(f"The requires_hyper_x is {hyper_x is not None}")
                x = m(x) 
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x


    def _profile_one_layer(self, m, x, dt): # seem not in use
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn): # add the detect_SCN part
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Detect_SCN)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

class DetectionModel_SCN(BaseModel_SCN):
    """YOLOv5 detection model class for object detection tasks, supporting custom configurations and anchors."""

    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None, hin=1):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.model, self.save, dimensions = parse_model_scn(deepcopy(self.yaml), ch=[ch])  # model, savelist

        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        self.hyper_stack = nn.Sequential( # hypernetwork not changing the width of the hypernetworks
            nn.Linear(hin, 64),
            nn.ReLU(),
            nn.Linear(64, dimensions), #
            nn.Softmax(dim=0) 
        )

        # Build strides, anchors
        m = self.model[-1]  # Detect()

        if isinstance(m, (Detect, Segment)): # 
            def _forward(x, hyper_x=None): # 
                """Passes the input 'x' through the model and returns the processed output."""
                # return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
                return self.forward(x, hyper_x)[0] if isinstance(m, Segment) else self.forward(x, hyper_x)

            def transform_angle(angle): # set alpha
                cos = math.cos(angle / 180 * math.pi)
                sin = math.sin(angle / 180 * math.pi)
                return Tensor([cos, sin]) #
            
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # check how to set hyper_x, for only to setup the hyper_x
            if hin == 1:
                hyper_out = Tensor([1.0]) # for scaling from 0.2-2.0
            elif hin == 2:
                hyper_out = transform_angle(180) # for translation

            # s = 256  # 2x min stride
            # m.inplace = self.inplace
            # m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s), hyper_out)])  # forward SCN check later
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once， try with open and not open

        if isinstance(m, Detect_SCN): # for all detect head 
            def _forward(x, hyper_x=None):
                # print("------------------------123123123123123-------------------")
                """Passes the input 'x' through the model and returns the processed output."""
                """Should always use the code that can cover the SCN cases"""
                return self.forward(x, hyper_x)[0] if isinstance(m, Segment) else self.forward(x, hyper_x)

            def transform_angle(angle): # set alpha
                cos = math.cos(angle / 180 * math.pi)
                sin = math.sin(angle / 180 * math.pi)
                return Tensor([cos, sin]) #
            
            s = 256  # 2x min stride
            m.inplace = self.inplace
            # check how to set hyper_x, for only to setup the hyper_x
            if hin == 1:
                hyper_out = Tensor([1.0]) # for scaling from 0.2-2.0
            elif hin == 2:
                hyper_out = transform_angle(180) # for translation
            
            # no need to make other modifications to change the results 
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s), hyper_out)])  # forward SCN check later
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1) # 这里为什么会出现相关的问题？
            self.stride = m.stride
            # done this in the conv2d with SCN bias defination
            # try if I dont do this
            self._initialize_biases()  # only run once to initialize bias

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, hyper_x=None, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        # if augment:
        #     return self._forward_augment(x)  # augmented inference, None when doing the testtime argumentation
        # return self._forward_once(x, profile, visualize)  # single-scale inference, train

        if hyper_x is not None: # 
            # print("ccccccccccc--------------------------------------------")
            hyper_out = self.hyper_stack(hyper_x) # hyper_x 
            if augment:
                return self._forward_augment_SCN(x, hyper_out)  # augmented inference 
            return self._forward_once(x, hyper_out, profile, visualize)  # single-scale inference, train
        else: #　only happen when checking the detect(head) layer
            # print("dddddddddddddddddd--------------------------------------------")
            if augment:
                return self._forward_augment(x)  # augmented inference, None
            return self._forward_once(x, profile=profile, visualize=profile)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
    
    def _forward_augment_SCN(self, x, hyper_x): # 
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi, hyper_x)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y): 
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    # def _initialize_biases(self, cf=None):
    #     """
    #     Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

    #     For details see https://arxiv.org/abs/1708.02002 section 3.3.
    #     """
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module 
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b.data[:, 5 : 5 + m.nc] += (
    #             math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
    #         )  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases(self, cf=None): # 
        """
        Initializes biases for YOLOv5's Detect_SCN module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        m = self.model[-1]  # Detect_SCN module
        if isinstance(m, Detect_SCN):
            for i, (bias_list, s) in enumerate(zip(m.conv_bias_list, m.stride)):
                for bias in bias_list:
                    b = bias.view(m.na, -1)  # reshape bias to (num_anchors, outputs_per_anchor)
                    b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                    b.data[:, 5:5 + m.nc] += (
                        math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
                    )  # cls
                    bias.data = b.view(-1)  # flatten back to 1D
        else:
            # Original initialization for non-SCN Detect modules
            for mi, s in zip(m.m, m.stride):
                b = mi.bias.view(m.na, -1)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)
                b.data[:, 5:5 + m.nc] += (
                    math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
                )
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

# # try another way with the abstract of SCN
# class SCN_YOLOv5s(SCN):
#     def __init__(self, num_alpha:int, dimensions:int) -> None:
#         base_model = DetectionModel(weights=None)

# Model_deteal = SCN_YOLOv5s

Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility
Model_SCN = DetectionModel_SCN  # retain YOLOv5 'Model' class for backwards compatibility , and make it possible 
# Model = DetectionModel_SCN  # retain YOLOv5 'Model' class for backwards compatibility , and make it possible 



class SegmentationModel(DetectionModel):
    """YOLOv5 segmentation model for object detection and segmentation tasks with configurable parameters."""

    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    """YOLOv5 classification model for image classification tasks, initialized with a config file or detection model."""

    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1

                
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def parse_model_scn(d, ch): # 
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )

    print("-----------------I am in the parse_model_scn")
    scn_dimensions = d.get("scn_dimensions", 0)  # default to 0
    print(f"------------------------scn_dimensions is {scn_dimensions}")

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m in { # all the model of the channel
            Conv_SCN, 
            Bottleneck_SCN, 
            C3_SCN, 
            SPPF_SCN,
            Conv_SCN_B,
        }: # 
            # extract channel 
            c1, c2 = ch[f], args[0]
            if c2 != no:  # 
                c2 = make_divisible(c2 * gw, ch_mul) # 

            args = [scn_dimensions, c1, c2, *args[1:]]
            if m is C3_SCN:
                args = [scn_dimensions, c1, c2, n] # OK with other stuff
                n = 1 # why?

        elif m is nn.BatchNorm2d: # 
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        # elif m in {Detect_SCN}: # might be OK here
        #     args.append([ch[x] for x in f])
        #     if isinstance(args[1], int):  # number of anchors
        #         args[1] = [list(range(args[1] * 2))] * len(f)
        #     if m is Segment:
        #         args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m in {Detect_SCN}: # set detection for SCN model
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f) # 
            # if m is Segment:
            #     args[3] = make_divisible(args[3] * gw, ch_mul)

            args = [scn_dimensions, *args]
            # print(f"The args is {args}")
            print("-----------------====================") #
            print(f"The args is {args}")
            print("-----------------====================") # 
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), scn_dimensions


if __name__ == "__main__": # 
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
