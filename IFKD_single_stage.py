# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from .single_stage import SingleStageDetector
import random
from math import sqrt
import torch.fft as fft
from PIL import Image
from mobile_sam import sam_model_registry,SamPredictor
model_type = "vit_t"
sam_checkpoint = "/media/yangxilab/DiskA/zhangs/mmdetection_ShipRSImagaeNet/mobile_sam/weights/mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

@MODELS.register_module()
class IFKDKDSingleStageDetector(SingleStageDetector):
    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        teacher_config: Union[ConfigType, str, Path],
        teacher_ckpt: Optional[str] = None,
        kd_cfg: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor)
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        self.teacher = MODELS.build(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        self.freeze(self.teacher)
        self.teacher_channels = kd_cfg['teacher_channels']
        self.generation = nn.Sequential(
            nn.Conv2d(self.teacher_channels, self.teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(self.teacher_channels, self.teacher_channels, kernel_size=3, padding=1))
        self.distill_cls_weight = 0.001
        self.distill_feat_weight = 0.0001
        self.stu_feature_adap = ADAP_SINGLE()
        self.complex_weight = nn.Parameter(torch.randn(1,256,2, dtype=torch.float32) * 0.02)
        self.mask_radio = 0.1
        self.p = 0.5
    def spectrum_noise(self, img_fft,M_frs = None):
        if random.random() > self.p:
            return img_fft
        B, C, H, W = img_fft.shape
        fft_shifted = fft.fftshift(img_fft, dim=(-2, -1))
        center_h, center_w = H//2, W//2
        max_radius = self.mask_radio * min(H, W) / 2
        u = torch.arange(H, device=img_fft.device)
        v = torch.arange(W, device=img_fft.device)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        dist_u = torch.abs(u_grid - center_h)
        dist_v = torch.abs(v_grid - center_w)
        max_dist = torch.maximum(dist_u, dist_v)
        M_binary = (max_dist <= max_radius).float().unsqueeze(0).unsqueeze(0)
        F_l = fft_shifted * M_binary         
        F_h = fft_shifted * (1 - M_binary)    
        F_l_reconstructed = F_l * M_frs[:,:,None,:]
        fft_shifted = F_l_reconstructed + F_h
        return fft.ifftshift(fft_shifted, dim=(-2, -1))
    def forward_FRD(self, x, M_frs = None):
        x = x.permute(0, 3, 1, 2)
        x_fft = fft.fft2(x, norm='ortho')
        x_fft = self.spectrum_noise(x_fft,M_frs)
        x = fft.ifft2(x_fft, norm='ortho')
        x = x.permute(0, 2, 3, 1) 
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight[:,:,None,None]
        return x.abs()
    def forward_EDE(self, x, img_metas):
        N, C, H, W = x.shape
        device = x.device
        images = torch.stack([
            Image.open(img["filename"]) for img in img_metas
        ]).cuda()
        sam_mask = mobile_sam(images,False)
        low_res_mask = torch.stack([x["masks"] for x in sam_mask], dim=0).squeeze(1).detach()
        sam_mask = F.interpolate(low_res_mask.float(), (x.shape[1],x.shape[2]), mode="bilinear", align_corners=False)
        masked_fea = torch.mul(x, sam_mask)
        new_fea = self.generation(masked_fea)
        return new_fea

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        
        x = self.extract_feat(img)
        y = self.teacher.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
        stu_feature_adap = self.stu_feature_adap(x)
        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]
        tea_bbox_outs = self.teacher.bbox_head(y)
        tea_cls_score = tea_bbox_outs[0]
        layers = len(stu_cls_score)
        distill_feat_loss, distill_cls_loss = 0, 0
        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            Frs_mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            Frs_mask = Frs_mask.detach()
            mask = HSM_dist(stu_feature_adap[layer],y[layer])
            mask = mask.detach()
            if(layer !=3 and layer!=4 and layer!=2):
                tea_feature =self.forward_FRD(y[layer],Frs_mask)
                stu_feature = self.forward_FRD(stu_feature_adap[layer],Frs_mask)
                feat_loss = torch.pow((tea_feature - stu_feature), 2)
            else :
                tea_feature =y[layer]
                stu_feature = stu_feature_adap[layer]
                feat_loss = torch.pow((self.forward_EDE(tea_feature)- stu_feature), 2)
            
            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid.detach(),reduction='none')
            distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum()
            distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()
        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight
          
        losses.update({"distill_feat_loss":distill_feat_loss})
        losses.update({"distill_cls_loss":distill_cls_loss})
        return losses
        
    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to cuda when calling ``cuda`` function."""
        self.teacher.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher is registered as a plain object, it is necessary to
        put the teacher model to other device when calling ``to`` function."""
        self.teacher.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        self.teacher.train(False)
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)


class ADAP_SINGLE(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 num = 5,
                 kernel = 3,
                 with_relu = False):
        super(ADAP_SINGLE, self).__init__()
        self.num = num
        self.with_relu = with_relu
        if kernel == 3:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=(1, 1))
        elif kernel == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            raise ValueError("other kernel size not completed")
        if with_relu:
            print("The adap conv is with relu")
        else:
            print("The adap conv is without relu")
        #self.const = nn.ConstantPad2d((0,1,0,1), 0.)
    def forward(self, inputs):
        out = []
        for i in range(self.num):
            if self.with_relu:
                out.append(F.relu(self.conv(inputs[i])))
            else:
                out.append(self.conv(inputs[i]))
        return out
    

def HSM_dist(x, y, *, c=1.0, keepdim=False):
    c_tensor = torch.as_tensor(c).type_as(x)
    sqrt_c = c_tensor ** 0.5
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    mobius_add = ((1 + 2 * c_tensor * xy + c_tensor * y2) * x + (1 - c_tensor * x2) * y) / \
                (1 + 2 * c_tensor * xy + c_tensor ** 2 * x2 * y2 + 1e-5)
    norm_term = (sqrt_c * (mobius_add - x).norm(dim=-1, p=2, keepdim=keepdim))
    dist_c = Artanh.apply(norm_term) 
    return dist_c * 2 / sqrt_c

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        clamped = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(clamped)
        return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))

    @staticmethod
    def backward(ctx, grad_output):
        (clamped,) = ctx.saved_tensors
        return grad_output / (1 - clamped.pow(2) + 1e-5)