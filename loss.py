import torch
from torch.nn import functional as F

from pet.models.ops import IOULoss
from pet.models.ops import SigmoidFocalLoss
from pet.utils.data.structures.boxlist_ops import boxlist_iou
from pet.utils.data.structures.boxlist_ops import cat_boxlist
from pet.rcnn.utils.matcher import Matcher
from pet.rcnn.utils.misc import cat, keep_only_positive_boxes
from pet.rcnn.core.config import cfg


def project_hier_to_fcos(hier, proposals, m):
    proposals = proposals.convert("xyxy")
    device = hier.hier.device
    boxes = proposals.bbox
    reg_targets = []
    _x1 = (hier.hier[:, :, 0] - boxes[:, 0, None]) * m / (boxes[:, 2, None] - boxes[:, 0, None])
    _y1 = (hier.hier[:, :, 1] - boxes[:, 1, None]) * m / (boxes[:, 3, None] - boxes[:, 1, None])
    _x2 = (hier.hier[:, :, 2] - boxes[:, 0, None]) * m / (boxes[:, 2, None] - boxes[:, 0, None])
    _y2 = (hier.hier[:, :, 3] - boxes[:, 1, None]) * m / (boxes[:, 3, None] - boxes[:, 1, None])
    new_hier = torch.stack([_x1, _y1, _x2, _y2], dim=-1)
    for _hier, box in zip(new_hier, boxes):
        x_step = 1.0
        y_step = 1.0
        shifts_x = torch.arange(
            0, m, step=x_step,
            dtype=torch.float32, device=device
        ) + x_step / 2
        shifts_y = torch.arange(
            0, m, step=y_step,
            dtype=torch.float32, device=device
        ) + y_step / 2
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        l = shift_x[:, None] - _hier[:, 0][None]
        t = shift_y[:, None] - _hier[:, 1][None]
        r = _hier[:, 2][None] - shift_x[:, None]
        b = _hier[:, 3][None] - shift_y[:, None]
        reg_targets_per_roi = torch.stack([l, t, r, b], dim=2)
        if cfg.HRCNN.CENTER_SAMPLE:
            inside_gt_bbox_mask = get_sample_region(_hier, x_step, y_step, shift_x, shift_y)
            reg_targets_per_roi[~inside_gt_bbox_mask] = -1
        reg_targets.append(reg_targets_per_roi)

    if proposals.bbox.shape[0] > 0:
        reg_targets = torch.cat(reg_targets, dim=0)

    return reg_targets


def get_sample_region(gt, x_step, y_step, shift_x, shift_y):
    num_gts = gt.shape[0]
    _num = gt.shape[1]
    K = len(shift_x)
    gt = gt[None].expand(K, num_gts, _num)
    center_x = (gt[..., 0] + gt[..., 2]) / 2
    center_y = (gt[..., 1] + gt[..., 3]) / 2
    center_gt = gt.new_zeros(gt.shape)

    radius = cfg.HRCNN.POS_RADIUS
    xmin = center_x[:] - x_step * radius
    ymin = center_y[:] - y_step * radius
    xmax = center_x[:] + x_step * radius
    ymax = center_y[:] + y_step * radius
    # limit sample region in gt
    center_gt[:, :, 0] = torch.where(xmin > gt[:, :, 0], xmin, gt[:, :, 0])
    center_gt[:, :, 1] = torch.where(ymin > gt[:, :, 1], ymin, gt[:, :, 1])
    center_gt[:, :, 2] = torch.where(xmax > gt[:, :, 2], gt[:, :, 2], xmax)
    center_gt[:, :, 3] = torch.where(ymax > gt[:, :, 3], gt[:, :, 3], ymax)

    left = shift_x[:, None] - center_gt[..., 0]
    right = center_gt[..., 2] - shift_x[:, None]
    top = shift_y[:, None] - center_gt[..., 1]
    bottom = center_gt[..., 3] - shift_y[:, None]
    center_bbox = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
    return inside_gt_bbox_mask


def _within_box(points, boxes):
    """Validate which hier are contained inside a given box.
    points: NxKx5
    boxes: Nx4
    output: NxK
    """
    x1_within = (points[..., 0] >= boxes[:, 0, None]) & (
            points[..., 0] <= boxes[:, 2, None]
    )
    y1_within = (points[..., 1] >= boxes[:, 1, None]) & (
            points[..., 1] <= boxes[:, 3, None]
    )
    x2_within = (points[..., 2] >= boxes[:, 0, None]) & (
            points[..., 2] <= boxes[:, 2, None]
    )
    y2_within = (points[..., 3] >= boxes[:, 1, None]) & (
            points[..., 3] <= boxes[:, 3, None]
    )

    return (x1_within & y1_within) | (x2_within & y2_within)


def center_within_box(points, boxes):
    """Validate which hier are contained inside a given box.
    points: NxKx5
    boxes: Nx4
    output: NxK
    """
    center_x = (points[..., 0] + points[..., 2]) / 2
    center_y = (points[..., 1] + points[..., 3]) / 2
    x_within = (center_x >= boxes[:, 0, None]) & (
            center_x <= boxes[:, 2, None]
    )
    y_within = (center_y >= boxes[:, 1, None]) & (
            center_y <= boxes[:, 3, None]
    )
    return x_within & y_within


class HierRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.roi_batch_size = cfg.HRCNN.ROI_BATCH_SIZE
        self.loss_weight = cfg.HRCNN.LOSS_WEIGHT
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.HRCNN.LOSS_GAMMA,
            cfg.HRCNN.LOSS_ALPHA
        )
        self.loc_loss_type = cfg.HRCNN.LOC_LOSS_TYPE
        self.box_reg_loss_func = IOULoss(self.loc_loss_type)
        self.centerness_loss_func = torch.nn.BCEWithLogitsLoss()

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Hier RCNN needs "labels" and "hier "fields for creating the targets
        target = target.copy_with_fields(["labels", "hier"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        positive_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            hier_per_image = matched_targets.get_field("hier")
            within_box = center_within_box(
                hier_per_image.hier, matched_targets.bbox
            )
            vis_hier = hier_per_image.hier[..., 4] > 0
            is_visible = (within_box & vis_hier).sum(1) > 0

            has_part = vis_hier[:, 2:].sum(1) == (within_box & vis_hier)[:, 2:].sum(1)
            is_visible = has_part & is_visible

            labels_per_image[~is_visible] = -1

            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            if self.roi_batch_size > 0:
                if self.roi_batch_size < positive_inds.shape[0]:
                    _inds = torch.randperm(positive_inds.shape[0])[:self.roi_batch_size]
                    positive_inds = positive_inds[_inds]

            proposals_per_image = proposals_per_image[positive_inds]
            hier_per_image = hier_per_image[positive_inds]
            proposals_per_image.add_field("hier_target", hier_per_image)
            positive_proposals.append(proposals_per_image)
        return positive_proposals

    def subsample(self, proposals, targets):
        positive_proposals = keep_only_positive_boxes(proposals)
        positive_proposals = self.prepare_targets(positive_proposals, targets)
        self.positive_proposals = positive_proposals

        all_num_positive_proposals = 0
        for positive_proposals_per_image in positive_proposals:
            all_num_positive_proposals += len(positive_proposals_per_image)
        if all_num_positive_proposals == 0:
            positive_proposals = [proposals[0][:1]]
        return positive_proposals

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, :, [0, 2]]
        top_bottom = reg_targets[:, :, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, hier_logits, box_regression=None, centerness=None):
        box_cls = hier_logits
        device = hier_logits.device
        reg_targets = []
        for proposals_per_image in self.positive_proposals:
            hier = proposals_per_image.get_field("hier_target")
            reg_targets_per_image = project_hier_to_fcos(
                hier, proposals_per_image, self.discretization_size
            )
            if len(reg_targets_per_image) > 0:
                reg_targets.append(reg_targets_per_image)

        if len(reg_targets) > 0:
            reg_targets = torch.cat(reg_targets, dim=0)
            labels = torch.zeros(reg_targets.shape[0], dtype=torch.int, device=device)

            pos_inds = reg_targets.min(dim=2)[0] > 0
            _pos_inds = pos_inds.max(dim=1)[0] > 0

            if _pos_inds.sum() > 0:
                reg_targets = reg_targets[_pos_inds]
                centerness_targets = self.compute_centerness_targets(reg_targets)
                centerness_targets[~pos_inds[_pos_inds]] = 0
                centerness_targets, max_id = centerness_targets.max(dim=1)
                reg_targets = reg_targets[range(len(max_id)), max_id]
                max_id += 1
                labels[_pos_inds] = max_id.int()

                num_classes = box_cls.size(1)

                box_cls_flatten = box_cls.permute(0, 2, 3, 1).reshape(-1, num_classes)
                box_regression_flatten = box_regression.permute(0, 2, 3, 1).reshape(-1, 4)
                centerness_flatten = centerness.permute(0, 2, 3, 1).reshape(-1)

                box_regression_flatten = box_regression_flatten[_pos_inds]
                centerness_flatten = centerness_flatten[_pos_inds]

                num_pos = torch.nonzero(_pos_inds > 0).squeeze(1)
                cls_loss = self.cls_loss_func(
                    box_cls_flatten,
                    labels
                ) / (num_pos.numel())
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets,
                    centerness_targets
                )
                centerness_loss = self.centerness_loss_func(
                    centerness_flatten,
                    centerness_targets
                )
                cls_loss *= self.loss_weight
                reg_loss *= self.loss_weight
                centerness_loss *= self.loss_weight
                return cls_loss, reg_loss, centerness_loss

        return box_cls.sum() * 0, box_regression.sum() * 0, centerness.sum() * 0


def hier_loss_evaluator():
    matcher = Matcher(
        cfg.HRCNN.FG_IOU_THRESHOLD,
        cfg.HRCNN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    resolution = cfg.HRCNN.RESOLUTION
    loss_evaluator = HierRCNNLossComputation(matcher, resolution)
    return loss_evaluator
