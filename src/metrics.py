import sklearn.metrics as skm


def calc_ap(precisions, recalls, interpolation_num=11):
    assert len(precisions) == len(recalls)

    # -------------- smooth out precision by the max right term --------------
    max_right_precision = 0  # fact: precision >= 0

    # smoothing the zigzag graph to a step graph
    for i in range(len(precisions) - 1, -1, -1):
        if precisions[i] <= max_right_precision:
            precisions[i] = max_right_precision
        else:
            max_right_precision = precisions[i]

    # ------------ mapping interpolation point to precision value ------------
    interpolation_points = [
        p / (interpolation_num - 1) for p in range(0, interpolation_num)
    ]

    # map interpolation point -> recalls index -> precisions index -> pr value
    interpolated_precision = list()
    last_mapped_ip_i = -1

    for i in range(len(recalls)):
        for j in range(last_mapped_ip_i + 1, len(interpolation_points)):
            if interpolation_points[j] <= recalls[i]:
                interpolated_precision.append(precisions[i])
                last_mapped_ip_i += 1
            else:
                break

    if last_mapped_ip_i + 1 < len(interpolation_points):
        for i in range(last_mapped_ip_i + 1, len(interpolation_points)):
            interpolated_precision.append(0)

    # print(interpolated_precision)
    # print(interpolation_points)

    # ------------ calculate AP_N where N = interpolation_num ------------
    return 1 / interpolation_num * sum(interpolated_precision)


def calc_aprf1(confusion_matrix):
    accuracy = calc_accuracy(confusion_matrix)

    precision = calc_precision(confusion_matrix)
    recall = calc_recall(confusion_matrix)

    f1 = calc_f1(precision, recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calc_f1(precision, recall):
    numerator = 2 * precision * recall
    denominator = precision + recall

    if numerator == 0:
        return 0

    return numerator / denominator


def calc_accuracy(confusion_matrix):  # list[list[int]]
    """
    calculate the accuracy value for the given confusion matrix
    :param confusion_matrix: assume to have format [[<TN>, <FP>], [<FN>, <TP>]]
    :return: <float> the accuracy value
    """
    true_negative, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]

    # if false_positive == false_negative == true_positive == 0:
    #     return 1

    numerator = true_positive + true_negative
    denominator = sum(confusion_matrix[0]) + sum(confusion_matrix[1])

    if numerator == 0:
        return 0

    return numerator / denominator


def calc_precision(confusion_matrix):  # list[list[int]]
    """
    calculate the precision value for the given confusion matrix
    :param confusion_matrix: assume to have format [[<TN>, <FP>], [<FN>, <TP>]]
    :return: <float> the precision value
    """
    _, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]

    # if false_positive == false_negative == true_positive == 0:
    #     return 1

    numerator = true_positive
    denominator = true_positive + false_positive

    if numerator == 0:
        return 0

    return numerator / denominator


def calc_recall(confusion_matrix):
    """
    calculate the recall value for the given confusion matrix
    :param confusion_matrix: assume to have format [[<TN>, <FP>], [<FN>, <TP>]]
    :return: <float> the recall value
    """
    _, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]

    # if false_positive == false_negative == true_positive == 0:
    #     return 1

    numerator = true_positive
    denominator = true_positive + false_negative

    if numerator == 0:
        return 0

    return numerator / denominator


def multilabel_cmatrix(pred_truth_label_map, t_confidence, t_iou):
    ious = pred_truth_label_map["iou_scores"]
    confis = pred_truth_label_map["confidence_scores"]
    preds = pred_truth_label_map["pred_labels"]
    truths = pred_truth_label_map["truth_labels"]

    preds_clabel = list()  # clabel is label for confusion matrix
    truths_clabel = list()
    for i in range(len(truths)):
        if not preds[i] or confis[i] < t_confidence:
            # if preds[i] exist, then confids[i] exist
            if truths[i]:
                preds_clabel.append("None")
                truths_clabel.append(truths[i])

        # after the above condition preds[i] must exist
        elif ious[i] and ious[i] >= t_iou:
            # if ious[i] exist, then preds[i] and truths[i] exist
            preds_clabel.append(preds[i])
            truths_clabel.append(truths[i])

        # elif ious[i] and ious[i] < t_iou:
        #     preds_clabel.append("None")
        #     truths_clabel.append(truths[i])

        else:  # only happen when truth label has the value of None
            preds_clabel.append(preds[i])
            truths_clabel.append("None")

    unique_labels = set(preds_clabel).union(set(truths_clabel))
    unique_labels.discard("None")

    unique_labels = ["None"] + list(unique_labels)
    _multilabel_cmatrix = skm.multilabel_confusion_matrix(
        y_true=truths_clabel,
        y_pred=preds_clabel,
        labels=unique_labels,
    ).astype(int)

    # print(len(ious), ious)
    # print(len(confis), confis)

    # print(len(truths), truths)
    # print(len(truths_clabel), truths_clabel)

    # print(len(preds), preds)
    # print(len(preds_clabel), preds_clabel)

    # print(unique_labels)

    class_cmatrix = dict()
    for i in range(1, len(unique_labels)):
        class_cmatrix[unique_labels[i]] = _multilabel_cmatrix[i]

    return class_cmatrix


def preds_choose_truths_map(truth_objs, pred_objs, iou_threshold=0):
    """
    This greedy method take a list of predictions assuming the predictions
    are sort from high confident to low confident in the list. For each
    prediction, the algorithm find the best fit truth box and store the
    t_box and p_box in a dict. It also made the box map to None if they
    not fit above the threshold.

    return: list of dict. The return list have the same order as the p_boxes
    arg (from prediction with the highest confident to lowest). The return
    dict in the list has form:
    {
        iou_score: <float> repr intersection over union of p_box and t_box
        pred_obj: <PredictClass> repr a pred object -- has the highest iou
            with the truth_obj compare to other truth obj
        truth_obj: <TruthClass> repr a truth object -- has the highest iou
            with the pred_obj compare to other pred obj
    }
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError(f"threshold must in [0, 1]")

    pred_truth_index_map = dict()
    pred_index_iou_map = dict()
    for i in range(len(pred_objs)):
        highest_iou = 0
        highest_truth_index = None

        for j in range(len(truth_objs)):
            iou_score = get_box_iou(
                t_box_cord=truth_objs[j].t_bbox,
                p_box_cord=pred_objs[i].p_bbox,
            )

            if iou_score > highest_iou:
                if j in pred_truth_index_map.values():
                    continue

                highest_iou = iou_score
                highest_truth_index = j

        # only map a pred to a truth if they overlap more than 0
        if highest_iou > iou_threshold:
            pred_truth_index_map[i] = highest_truth_index
            pred_index_iou_map[i] = highest_iou

    pred_truth_map = list()
    for i in range(len(pred_objs)):
        if i in pred_truth_index_map.keys():
            pred_truth_map.append(
                {
                    "iou_score": pred_index_iou_map[i],
                    "pred_obj": pred_objs[i],
                    "truth_obj": truth_objs[pred_truth_index_map[i]],
                }
            )
        else:
            pred_truth_map.append(
                {
                    "iou_score": None,
                    "pred_obj": pred_objs[i],
                    "truth_obj": None,
                }
            )

    for j in range(len(truth_objs)):
        if j not in pred_truth_index_map.values():
            pred_truth_map.append(
                {
                    "iou_score": None,
                    "pred_obj": None,
                    "truth_obj": truth_objs[j],
                }
            )

    # from pprint import pprint
    # pprint(pred_truth_map)
    return pred_truth_map


def get_box_iou(t_box_cord, p_box_cord):
    p_box = {
        "x1": p_box_cord[0],
        "y1": p_box_cord[1],
        "x2": p_box_cord[2],
        "y2": p_box_cord[3],
    }

    t_box = {
        "x1": t_box_cord[0],
        "y1": t_box_cord[1],
        "x2": t_box_cord[2],
        "y2": t_box_cord[3],
    }

    if p_box["x1"] > p_box["x2"] or p_box["y1"] > p_box["y2"]:
        raise ValueError("p_box_cord must in (xmin, ymin, xmax, ymax) format")

    if t_box["x1"] > t_box["x2"] or t_box["y1"] > t_box["y2"]:
        raise ValueError("t_box_cord must in (xmin, ymin, xmax, ymax) format")

    # coordinate of the intersection rectangle
    inter_x1 = max(p_box["x1"], t_box["x1"])
    inter_y1 = max(p_box["y1"], t_box["y1"])
    inter_x2 = min(p_box["x2"], t_box["x2"])
    inter_y2 = min(p_box["y2"], t_box["y2"])

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0

    # area of the intersection rectangle
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # area of predicted box and truth box
    p_box_area = (p_box["x2"] - p_box["x1"]) * (p_box["y2"] - p_box["y1"])
    t_box_area = (t_box["x2"] - t_box["x1"]) * (t_box["y2"] - t_box["y1"])

    return intersection_area / (p_box_area + t_box_area - intersection_area)
