    net_img_size = [roi.size for roi in rois]
    all_part_boxes = [[] for _ in range(len(rois))]
    all_hier_boxes = [[] for _ in range(len(rois))]
    all_hier_scores = [[] for _ in range(len(rois))]
    aug_idx = 0

    conv_features = features[aug_idx][1]
    results = model.hier_net(conv_features, rois, targets=None)

    if cfg.TEST.BBOX_AUG.ENABLED and cfg.TEST.HIER_AUG.ENABLED:
        if len(rois[0]) == 0:
            return results
        part_boxes = [result.get_field("part_boxes") for result in results]
        add_results(all_part_boxes, part_boxes)
        if cfg.HRCNN.EVAL_HIER:
            hier_boxes = [result.get_field("hier_boxes") for result in results]
            add_results(all_hier_boxes, hier_boxes)
            hier_scores = [result.get_field("hier_scores") for result in results]
            add_results(all_hier_scores, hier_scores)
        aug_idx += 1

        if cfg.TEST.BBOX_AUG.H_FLIP:
            rois_hf = [roi.transpose(0) for roi in rois]
            features_hf = features[aug_idx][1]
            results_hf = model.hier_net(features_hf, rois_hf, targets=None)
            part_boxes_hf = [result_hf.get_field("part_boxes") for result_hf in results_hf]
            part_boxes_hf = [flip_hier_box_result(_part_boxes_hf) for _part_boxes_hf in part_boxes_hf]
            add_results(all_part_boxes, part_boxes_hf)
            if cfg.HRCNN.EVAL_HIER:
                for i in range(len(rois)):
                    boxes_hf, score_hf = process_hier_result(results_hf[i].get_field("hier_boxes"),
                                                             results_hf[i].get_field("hier_scores"),
                                                             features[aug_idx][0][i], flip=True)
                    all_hier_boxes[i].append(boxes_hf)
                    all_hier_scores[i].append(score_hf)
            aug_idx += 1

        for scale in cfg.TEST.BBOX_AUG.SCALES:
            rois_scl = [roi.resize(size) for roi, size in zip(rois, features[aug_idx][0])]
            features_scl = features[aug_idx][1]
            results_scl = model.hier_net(features_scl, rois_scl, targets=None)
            part_boxes_scl = [result_scl.get_field("part_boxes") for result_scl in results_scl]
            part_boxes_scl = [_part_boxes_scl.resize(size) for _part_boxes_scl, size
                              in zip(part_boxes_scl, net_img_size)]
            add_results(all_part_boxes, part_boxes_scl)
            if cfg.HRCNN.EVAL_HIER:
                for i in range(len(rois)):
                    boxes_scl, score_scl = process_hier_result(results_scl[i].get_field("hier_boxes"),
                                                               results_scl[i].get_field("hier_scores"),
                                                               features[aug_idx][0][i], net_img_size[i])
                    all_hier_boxes[i].append(boxes_scl)
                    all_hier_scores[i].append(score_scl)
            aug_idx += 1

            if cfg.TEST.BBOX_AUG.H_FLIP:
                rois_scl_hf = [roi.resize(size) for roi, size in zip(rois, features[aug_idx][0])]
                rois_scl_hf = [roi.transpose(0) for roi in rois_scl_hf]
                features_scl_hf = features[aug_idx][1]
                results_scl_hf = model.hier_net(features_scl_hf, rois_scl_hf, targets=None)
                part_boxes_scl_hf = [result_scl_hf.get_field("part_boxes") for result_scl_hf in results_scl_hf]
                part_boxes_scl_hf = [flip_hier_box_result(_part_boxes_scl_hf)
                                     for _part_boxes_scl_hf in part_boxes_scl_hf]
                part_boxes_scl_hf = [_part_boxes_scl_hf.resize(size) for _part_boxes_scl_hf, size
                                     in zip(part_boxes_scl_hf, net_img_size)]
                add_results(all_part_boxes, part_boxes_scl_hf)

                if cfg.HRCNN.EVAL_HIER:
                    for i in range(len(rois)):
                        boxes_scl_hf, score_scl_hf = process_hier_result(results_scl_hf[i].get_field("hier_boxes"),
                                                                         results_scl_hf[i].get_field("hier_scores"),
                                                                         features[aug_idx][0][i], net_img_size[i],
                                                                         flip=True)
                        all_hier_boxes[i].append(boxes_scl_hf)
                        all_hier_scores[i].append(score_scl_hf)
                aug_idx += 1

        nms_thresh, detections_per_img = get_detection_params()
        for result, part_boxes in zip(results, all_part_boxes):
            part_boxes = filter_results(
                cat_boxlist(part_boxes), nms_thresh=nms_thresh, detections_per_img=detections_per_img
            )
            result.add_field("part_boxes", part_boxes)
        if cfg.HRCNN.EVAL_HIER:
            for hier_boxes, hier_scores, result in zip(all_hier_boxes, all_hier_scores, results):
                # TODO
                # hier_boxes = torch.stack(hier_boxes, dim=0).mean(0)
                # hier_scores = torch.stack(hier_scores, dim=0).mean(0)
                # Multi-scale test for hier is not work, so we use last scale results as final results.
                i = -2 if cfg.TEST.BBOX_AUG.H_FLIP else -1
                hier_boxes = hier_boxes[i]
                hier_scores = hier_scores[i]
                result.add_field("hier_boxes", hier_boxes)
                result.add_field("hier_scores", hier_scores)
