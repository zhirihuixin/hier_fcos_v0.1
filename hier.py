    # hier r-cnn (different from faster r-cnn)
    if cfg.MODEL.HIER_ON:
        nms_thresh = cfg.HRCNN.NMS_TH
        detections_per_img = cfg.HRCNN.DETECTIONS_PER_IMG
 



def flip_hier_box_result(box_result):
    FLIP_MAP = ([4, 5], [6, 7])
    box_result = box_result.transpose(0)
    labels = box_result.get_field("labels")
    for j in FLIP_MAP:
        l_idx = labels == j[0]
        r_idx = labels == j[1]
        labels[l_idx] = j[1]
        labels[r_idx] = j[0]
    box_result.add_field("labels", labels)

    return box_result


def process_hier_result(boxes, scores, size, ori_size=None, flip=False):
    FLIP_MAP = ([2, 3], [4, 5])
    if flip:
        boxes_flip = boxes.clone()
        boxes_flip[:, :, 0] = size[0] - boxes[:, :, 2] - 1
        boxes_flip[:, :, 2] = size[0] - boxes[:, :, 0] - 1
        idx = torch.arange(6, device=scores.device)
        for j in FLIP_MAP:
            idx[j[0]] = j[1]
            idx[j[1]] = j[0]
        boxes = boxes_flip[:, idx]
        scores = scores[:, idx]
    if ori_size is not None:
        boxes[:, :, 0:4:2] = boxes[:, :, 0:4:2] * ori_size[0] / size[0]
        boxes[:, :, 1:4:2] = boxes[:, :, 1:4:2] * ori_size[1] / size[1]
    return boxes, scores
