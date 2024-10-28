# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torchvision

import imgaug as ia
from imgaug import augmenters as iaa

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy, xyxy2xywhn
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    """Provides optional data augmentation for YOLOv5 using Albumentations library if installed."""

    def __init__(self, size=640):
        """Initializes Albumentations class for optional data augmentation in YOLOv5 with specified input size."""
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        """Applies transformations to an image and labels with probability `p`, returning updated image and labels."""
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False): 
    """
    Applies ImageNet normalization to RGB images in BCHW format, modifying them in-place if specified.

    Example: y = (x - mean) / std
    """
    return TF.normalize(x, mean, std, inplace=inplace) 



def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD): 
    """Reverses ImageNet normalization for BCHW format RGB images by applying `x = x * std + mean`."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """Applies HSV color-space augmentation to an image with random gains for hue, saturation, and value."""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """Equalizes image histogram, with optional CLAHE, for BGR or RGB image with shape (n,m,3) and range 0-255."""
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """
    Replicates half of the smallest object labels in an image for data augmentation.

    Returns augmented image and labels.
    """
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# will be used in every epoch
# I should still change this part to train rotation, I guess the scaling is OK
# 
def random_perspective( 
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    """Applies random perspective transformation to an image, modifying the image and corresponding labels."""
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2 # 

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations # why?
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    # 
    # print(f"The value of the angle is {a} at the batch ")
    # if a < 0.5 and a > -0.5:
    #     print(f"The value of the angle is {a} at the batch ")
    # if a < 90.5 and a > 89.5:
    #     print(f"The value of the angle is {a} at the batch")
    # if a > -90.5 and a < -89.5:
    #     print(f"The value of the angle is {a} at the batch ")

    # if a > 179.5: # 
    #     print(f"The value of the angle is {a} at the batch ")
    # if a < -179.5:
    #     print(f"The value of the angle is {a} at the batch ")

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix 
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    # print(f"The target is {targets}")
    # print(f"The size and shape is {targets.shape}") # 
    # print(f"type is {type(targets)}")
    # return _
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


# Try with this one  
def fixed_perspective_batch( # 
    imgs, targets=(), degrees=0, translate=0, scale=1
):
    """Applies random perspective transformation to a batch of images, modifying the images and corresponding labels."""
    
    batch_size = imgs.shape[0]
    height = imgs.shape[2]
    width = imgs.shape[3] 

    # image_shape = imgs.numpy().shape[]
    image_shape = imgs[0].permute(1, 2, 0).numpy().shape # OK 
    # print(f"The shape is {image_shape}") 

    seq = iaa.Sequential([
        # iaa.Flipud(0.5),  # 对50%的图像做上下翻转
        # iaa.Fliplr(0.5),  # 对50%的图像做左右翻转
        # # iaa.Multiply((1.2, 1.5)),  # 像素乘上1.2或者1.5之间的数字
        # # iaa.GaussianBlur(sigma=(0, 3.0)),  # 使用高斯模糊
        # iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
        #                    children=iaa.WithChannels(0, iaa.Add(10))),  # 先将图片从RGB变换到HSV,然后将H值增加10,然后再变换回RGB。
        iaa.Affine( 
            # translate_px={"x": 15, "y": 15},  # 平移
            scale=scale,  
            rotate=degrees 
        )  # 仿射变换
    ])

    seq_det = seq.to_deterministic()

    targets_xywhn = targets[:, 2:] # 
    # print(f"The normalized_xywh is {targets_xywhn}") 

    target_item_size = targets.size(0) # how many size that the 
    # print(f"The number of teh target_item_size is {target_item_size}")
    multipler = torch.tensor([width, height, width, height]) # original size 

    origin_xyxy = xywhn2xyxy(targets_xywhn, width, height)

    # print(f"The origin_xyxy is {origin_xyxy}") 
    # print(f"The origin_xyxy.numpy is {origin_xyxy.numpy()}") 

    bbs = ia.BoundingBoxesOnImage([ 
        ia.BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for b in origin_xyxy.numpy()
    ], shape=image_shape) 

    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # targets_xywh = targets_xywh*multipler

    new_bndbox_list = []

    for bbox in bbs_aug.bounding_boxes:
        new_bndbox_list.append([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
    
    # Prepare output lists
    transformed_imgs = [] 
    updated_targets = []

    # print(f"new_bndbox_list is {new_bndbox_list}")

    new_xyxy = torch.tensor(new_bndbox_list)

    # 或者从 NumPy 数组转换为张量
    # new_bndbox_tensor_from_np = torch.tensor(np.array(new_bndbox_tensor))

    # 打印张量和它的形状
    # print(new_xyxy)
    # print(new_xyxy.shape) 

    new_xywhn = xyxy2xywhn(new_xyxy, width, height) # 

    # print(f"new_normalized bbox is {new_xywhn}") 
    # print(f"new_normalized bbox shape is {new_xywhn.shape}") 

    targets[:, 2:6] = new_xywhn # 
    
    return transformed_imgs, targets 

# Try with this one  
# might be in low performance
# basically following the same idea as the v1, but the code is more concise and efficient 
def fixed_perspective_batch_v2( # 
    imgs, targets=(), degrees=0, translate=0, scale=1
):
    """Applies random perspective transformation to a batch of images, modifying the images and corresponding labels."""
    
    batch_size = imgs.shape[0] # 
    height = imgs.shape[2]
    width = imgs.shape[3] 

    height_bias = height//2
    width_bias = width//2

    # print(f"The width_bias and height_bias are {width_bias} and {height_bias}")

    # image_shape = imgs.numpy().shape[]
    image_shape = imgs[0].permute(1, 2, 0).numpy().shape # OK 
    # print(f"The shape is {image_shape}") 

    targets_xywhn = targets[:, 2:] # 
    # print(f"The shape of the labels is{targets_xywhn.shape}")
    # print(f"The normalized_xywh is {targets_xywhn}") 

    target_item_size = targets.size(0) # how many size that the 
    # print(f"The number of teh target_item_size is {target_item_size}")
    multipler = torch.tensor([width, height, width, height]) # original size 

    origin_xyxy = xywhn2xyxy(targets_xywhn, width, height)

    # print(f"The origin_xyxy is {origin_xyxy}") 
    # print(f"The origin_xyxy.numpy is {origin_xyxy.numpy()}") 

    x_coords = origin_xyxy[:, [0, 2, 0, 2]] - width_bias  # 选择 x1 和 x2 (第 1 列和第 3 列)
    y_coords = origin_xyxy[:, [1, 1, 3, 3]] - height_bias  # 选择 y1 和 y2 (第 2 列和第 4 列)

    # print(f"The x coordinate for each bbox is {x_coords}")
    # print(f"The y coordinate for each bbox is {y_coords}")

    r = torch.sqrt(x_coords**2 + y_coords**2)  # r = √(x^2 + y^2)
    theta = torch.atan2(y_coords, x_coords)    # θ = arctan(y/x)

    # print(f"r: {r}")
    # print(f"theta: {theta}")

    angle_to_theta = math.radians(degrees)

    theta += angle_to_theta # -: means clockwise  +: means anti-clockwise

    # print(f"r: {r}")
    # print(f"theta: {theta}")

    x_cartesian = r * torch.cos(theta) + width_bias # 这里的内容需要重新更新
    y_cartesian = r * torch.sin(theta) + height_bias

    # print(f"The x coordinate after process for each bbox is {x_cartesian}")
    # print(f"The y coordinate after process for each bbox is {y_cartesian}")

    x_cartesian = torch.clamp(x_cartesian, 0, width) #
    y_cartesian = torch.clamp(y_cartesian, 0, height) #

    # print(f"new_normalized bbox is {new_xywhn}") 
    # print(f"new_normalized bbox shape is {new_xywhn.shape}") 

    x_min, _ = torch.min(x_cartesian, dim=1) # x_min
    x_max, _ = torch.max(x_cartesian, dim=1) # x_max
    y_min, _ = torch.min(y_cartesian, dim=1) # y_min
    y_max, _ = torch.max(y_cartesian, dim=1) # y_max

    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    # print(f"The bboxes is {bboxes} and the size is {bboxes.shape}")

    new_xywhn = xyxy2xywhn(bboxes, width, height) # 

    # print(f"The new_xywhn is {new_xywhn}") 

    targets[:, 2:6] = new_xywhn # 
    
    return None, targets 

# Try with this one  
# random rotation for different images in a batch
# not OK currently for the 
# imgs, targets=(), degrees=0, translate=0, scale=1
def fixed_perspective_batch_v2a( # 
    ims, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=0, perspective=0.0, border=(0, 0)
):
    """Applies random perspective transformation to a batch of images, modifying the images and corresponding labels."""
    batch_size = ims.shape[0]  # 获取批次大小
    height = ims.shape[2] + border[0] * 2  # shape(N,C,H,W) -> height
    width = ims.shape[3] + border[1] * 2  # shape(N,C,H,W) -> width

    # 初始化输出
    # 8bit inputs
    transformed_images = np.zeros((batch_size, height, width, ims.shape[1]), dtype=np.uint8)  # 修改通道数为C
    new_targets = torch.empty(0, 6) # set an empty tensor
    
    # 先看看target是怎么样的
    # print(f"The target is {targets}")
    # print(f"The shape of the target is {targets.shape}")

    for b in range(batch_size):
        # im = ims[b].transpose(1, 2, 0)  # 将 (C, H, W) 转换为 (H, W, C)
        im = ims[b].permute(1, 2, 0)  # 将 (C, H, W) 转换为 (H, W, C)
        im = im.cpu().numpy()  # 确保 im 是 NumPy 数组



        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix 
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        # 应用变换
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                transformed_images[b] = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                transformed_images[b] = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # 上面调通了，剩下的就是下面的了
        # Transform label coordinates

        target_in_batch = targets[targets[:, 0] == b]  # 选择第二列等于 b 的行
        print(f"The target is {target_in_batch}")
        print(f"The shape of the target is {target_in_batch.shape}")

        if len(target_in_batch) > 0: # 针对当前batch
            n = len(target_in_batch)
            print(f"The size of the target_in_batch is {n}")
            use_segments = any(x.any() for x in segments) and len(segments) == n
            new = np.zeros((n, 4))  # 更新为5列，包含类别和边界框

            if use_segments:  # warp segments
                segments = resample_segments(segments)  # upsample
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
                    new[i, 1:5] = segment2box(xy, width, height)  # 更新边界框
                    new[i, 0] = targets[i, 0]  # 类别标签
                    new[i, 5] = b  # 更新为当前图像的索引

            else:  # warp boxes
                # 这里需要进行一次转换，转换成
                targets_in_batch_np = target_in_batch[:, 2:6]
                print(f"The extracted type of the name is {targets_in_batch_np}")
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new[:, 1:5] = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
                new[:, 0] = targets[:, 0]  # 类别标签
                new[:, 5] = b  # 更新为当前图像的索引

        #     # filter candidates
        #     i = box_candidates(box1=targets[:, 1:5].T * s, box2=new[:, 1:5].T, area_thr=0.01 if use_segments else 0.10)
        #     targets = new[i]  # 更新为新的目标
        #     targets[:, 1:5] = new[i, 1:5]  # 更新边界框
        exit()

    # final operations for the images and targets to get the conversion
    transformed_images = transformed_images.transpose(0, 3, 1, 2)  # Change from (N, H, W, C) to (N, C, H, W)
    # transformed_images = tf.convert_to_tensor(transformed_images)  # 将 NumPy 数组转换为 TensorFlow tensor
    transformed_images = torch.from_numpy(transformed_images)  # 将 NumPy 数组转换为 PyTorch tensor

    # 到时候输出来看看就指知道了

    return transformed_images, targets

# for only scaling, but with the same principle with the 
def fixed_perspective_batch_v3( 
    imgs, targets=(), degrees=0, translate=0, scale=1
):
    """Applies random perspective transformation to a batch of images, modifying the images and corresponding labels."""
    
    """Applies scaling transformation to a batch of images, modifying the images and corresponding labels."""
    
    batch_size = imgs.shape[0]
    height = imgs.shape[2]
    width = imgs.shape[3]

    height_bias = height // 2
    width_bias = width // 2

    # image_shape = imgs.numpy().shape[]
    image_shape = imgs[0].permute(1, 2, 0).numpy().shape

    targets_xywhn = targets[:, 2:]

    target_item_size = targets.size(0)
    multipler = torch.tensor([width, height, width, height])

    origin_xyxy = xywhn2xyxy(targets_xywhn, width, height)

    x_coords = origin_xyxy[:, [0, 2, 0, 2]] - width_bias
    y_coords = origin_xyxy[:, [1, 1, 3, 3]] - height_bias

    # Calculate the new coordinates based on scaling
    x_scaled = x_coords * scale + width_bias
    y_scaled = y_coords * scale + height_bias

    # Clip the scaled coordinates to be within the image boundaries
    x_scaled = torch.clamp(x_scaled, 0, width)
    y_scaled = torch.clamp(y_scaled, 0, height)

    x_min, _ = torch.min(x_scaled, dim=1)
    x_max, _ = torch.max(x_scaled, dim=1)
    y_min, _ = torch.min(y_scaled, dim=1)
    y_max, _ = torch.max(y_scaled, dim=1)

    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    new_xywhn = xyxy2xywhn(bboxes, width, height)

    targets[:, 2:6] = new_xywhn
    
    return None, targets

def copy_paste(im, labels, segments, p=0.5): 
    """
    Applies Copy-Paste augmentation by flipping and merging segments and labels on an image.

    Details at https://arxiv.org/abs/2012.07177.
    """
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5): 
    """
    Applies cutout augmentation to an image with optional label adjustment, using random masks of varying sizes.

    Details at https://arxiv.org/abs/1708.04552.
    """
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """
    Applies MixUp augmentation by blending images and labels.

    See https://arxiv.org/pdf/1710.09412.pdf for details.
    """
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """
    Filters bounding box candidates by minimum width-height threshold `wh_thr` (pixels), aspect ratio threshold
    `ar_thr`, and area ratio threshold `area_thr`.

    box1(4,n) is before augmentation, box2(4,n) is after augmentation.
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    """Sets up and returns Albumentations transforms for YOLOv5 classification tasks depending on augmentation
    settings.
    """
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """Applies a series of transformations including center crop, ToTensor, and normalization for classification."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    """Resizes and pads images to specified dimensions while maintaining aspect ratio for YOLOv5 preprocessing."""

    def __init__(self, size=(640, 640), auto=False, stride=32):
        """Initializes a LetterBox object for YOLOv5 image preprocessing with optional auto sizing and stride
        adjustment.
        """
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):
        """
        Resizes and pads input image `im` (HWC format) to specified dimensions, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    """Applies center crop to an image, resizing it to the specified size while maintaining aspect ratio."""

    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center crop to the input image and resizes it to a specified size, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    """Converts BGR np.array image from HWC to RGB CHW format, normalizes to [0, 1], and supports FP16 if half=True."""

    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
