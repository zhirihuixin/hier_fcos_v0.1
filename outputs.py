import math

import torch
from torch import nn
from torch.nn import functional as F

from pet.utils.net import make_conv
from pet.models.ops import ConvTranspose2d
from pet.models.ops import interpolate
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


@registry.ROI_HIER_OUTPUTS.register("Hier_output")
class Hier_output(nn.Module):
    def __init__(self, dim_in):
        super(Hier_output, self).__init__()

        num_classes = cfg.HRCNN.NUM_CLASSES
        num_convs = cfg.HRCNN.OUTPUT_NUM_CONVS
        conv_dim = cfg.HRCNN.OUTPUT_CONV_DIM
        use_lite = cfg.HRCNN.OUTPUT_USE_LITE
        use_bn = cfg.HRCNN.OUTPUT_USE_BN
        use_gn = cfg.HRCNN.OUTPUT_USE_GN
        use_dcn = cfg.HRCNN.OUTPUT_USE_DCN

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            conv_type = 'deform' if use_dcn and i == num_convs - 1 else 'normal'
            cls_tower.append(
                make_conv(dim_in, conv_dim, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )
            bbox_tower.append(
                make_conv(dim_in, conv_dim, kernel=3, stride=1, dilation=1, use_dwconv=use_lite,
                          conv_type=conv_type, use_bn=use_bn, use_gn=use_gn, use_relu=True, kaiming_init=False,
                          suffix_1x1=use_lite)
            )
            dim_in = conv_dim

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            conv_dim, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            conv_dim, 4, kernel_size=3, stride=1, padding=1
        )
        self.centerness = nn.Conv2d(
            conv_dim, 1, kernel_size=3, stride=1, padding=1
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.HRCNN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        box_tower = self.bbox_tower(x)

        logits = self.cls_logits(cls_tower)
        if cfg.HRCNN.CENTERNESS_ON_REG:
            centerness = self.centerness(box_tower)
        else:
            centerness = self.centerness(cls_tower)
        if cfg.HRCNN.NORM_REG_TARGETS:
            bbox_reg = F.relu(self.bbox_pred(box_tower))
        else:
            bbox_reg = torch.exp(self.bbox_pred(box_tower))

        return logits, bbox_reg, centerness
