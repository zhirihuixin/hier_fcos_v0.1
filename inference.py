import numpy as np
import cv2

import torch
from torch import nn

from pet.utils.data.structures.bounding_box import BoxList
from pet.utils.data.structures.boxlist_ops import cat_boxlist
from pet.utils.data.structures.boxlist_ops import remove_small_boxes
from pet.rcnn.core.config import cfg


class HierPostProcessor(nn.Module):
    def __init__(self):
        super(HierPostProcessor, self).__init__()
        self.num_classes = cfg.HRCNN.NUM_CLASSES
        self.m = cfg.HRCNN.RESOLUTION

    def forward(self, box_cls_all, box_reg_all, centerness_all, boxes_all):
        device = box_cls_all.device
        boxes_per_image = [len(box) for box in boxes_all]
        cls = box_cls_all.split(boxes_per_image, dim=0)
        reg = box_reg_all.split(boxes_per_image, dim=0)
        center = centerness_all.split(boxes_per_image, dim=0)

        results = []
        for box_cls, box_regression, centerness, boxes in zip(cls, reg, center, boxes_all):
            N, C, H, W = box_cls.shape
            # put in the same format as locations
            box_cls = box_cls.permute(0, 2, 3, 1).reshape(N, -1, self.num_classes).sigmoid()
            box_regression = box_regression.permute(0, 2, 3, 1).reshape(N, -1, 4)
            centerness = centerness.permute(0, 2, 3, 1).reshape(N, -1).sigmoid()

            # multiply the classification scores with centerness scores
            box_cls = box_cls * centerness[:, :, None]
            _boxes = boxes.bbox
            size = boxes.size
            boxes_scores = boxes.get_field("scores")
            results_per_image = [boxes]
            for i in range(N):
                box = _boxes[i]
                boxes_score = boxes_scores[i]
                per_box_cls = box_cls[i]
                per_box_cls_max, per_box_cls_inds = per_box_cls.max(dim=0)

                per_class = torch.range(2, 1 + self.num_classes, dtype=torch.long, device=device)

                per_box_regression = box_regression[i]
                per_box_regression = per_box_regression[per_box_cls_inds]

                x_step = 1.0
                y_step = 1.0
                shifts_x = torch.arange(
                    0, self.m, step=x_step,
                    dtype=torch.float32, device=device
                ) + x_step / 2
                shifts_y = torch.arange(
                    0, self.m, step=y_step,
                    dtype=torch.float32, device=device
                ) + y_step / 2
                shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
                shift_x = shift_x.reshape(-1)
                shift_y = shift_y.reshape(-1)
                locations = torch.stack((shift_x, shift_y), dim=1)
                per_locations = locations[per_box_cls_inds]

                _x1 = per_locations[:, 0] - per_box_regression[:, 0]
                _y1 = per_locations[:, 1] - per_box_regression[:, 1]
                _x2 = per_locations[:, 0] + per_box_regression[:, 2]
                _y2 = per_locations[:, 1] + per_box_regression[:, 3]

                _x1 = _x1 / self.m * (box[2] - box[0]) + box[0]
                _y1 = _y1 / self.m * (box[3] - box[1]) + box[1]
                _x2 = _x2 / self.m * (box[2] - box[0]) + box[0]
                _y2 = _y2 / self.m * (box[3] - box[1]) + box[1]

                detections = torch.stack([_x1, _y1, _x2, _y2], dim=-1)

                boxlist = BoxList(detections, size, mode="xyxy")
                boxlist.add_field("labels", per_class)
                boxlist.add_field("scores", torch.sqrt(torch.sqrt(per_box_cls_max) * boxes_score))
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = remove_small_boxes(boxlist, 0)
                results_per_image.append(boxlist)

            results_per_image = cat_boxlist(results_per_image)
            results.append(results_per_image)

        return results


def hier_post_processor():
    hier_post_processor = HierPostProcessor()
    return hier_post_processor
