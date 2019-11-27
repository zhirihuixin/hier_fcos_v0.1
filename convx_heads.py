from torch import nn
from torch.nn import functional as F

from pet.utils.net import make_conv
from pet.rcnn.utils.poolers import Pooler
from pet.rcnn.modeling import registry
from pet.rcnn.core.config import cfg


@registry.ROI_HIER_HEADS.register("roi_convx_head")
class roi_convx_head(nn.Module):
    def __init__(self, dim_in, spatial_scale):
        super(roi_convx_head, self).__init__()
        self.dim_in = dim_in[-1]

        resolution = cfg.HRCNN.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.HRCNN.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=spatial_scale,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_lite = cfg.HRCNN.CONVX_HEAD.USE_LITE
        use_bn = cfg.HRCNN.CONVX_HEAD.USE_BN
        use_gn = cfg.HRCNN.CONVX_HEAD.USE_GN
        conv_dim = cfg.HRCNN.CONVX_HEAD.CONV_DIM
        num_stacked_convs = cfg.HRCNN.CONVX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.HRCNN.CONVX_HEAD.DILATION

        self.blocks = []
        for layer_idx in range(num_stacked_convs):
            layer_name = "hier_fcn{}".format(layer_idx + 1)
            module = make_conv(self.dim_in, conv_dim, kernel=3, stride=1, dilation=dilation, use_dwconv=use_lite,
                               use_bn=use_bn, use_gn=use_gn, suffix_1x1=use_lite)
            self.add_module(layer_name, module)
            self.dim_in = conv_dim
            self.blocks.append(layer_name)
        self.dim_out = self.dim_in

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x
