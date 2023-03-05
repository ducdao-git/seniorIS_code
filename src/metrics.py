from truth_class import TruthClass
from predict_class import PredictClass


def preds_choose_truths_map(
        truth_objs: list[TruthClass], pred_objs: list[PredictClass],
        threshold=0.5
):
    """
    This greedy method take a list of predictions assuming the predictions
    are sort from high confident to low confident in the list. For each
    prediction, the algorithm find the best fit truth box and store the
    t_box and p_box in a dict. It also made the box map to background if they
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
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold must in (0, 1]")

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

        if highest_iou > threshold:
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

# assert get_box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1
# assert get_box_iou([1, 0, 2, 2], [0, 0, 2, 2]) == 0.5
# assert get_box_iou([1, 1, 2, 2], [0, 0, 2, 2]) == 0.25
#
# assert get_box_iou([0, 1, 2, 2], [0, 0, 1, 1]) == 0
# assert get_box_iou([2, 2, 3, 3], [0, 0, 1, 1]) == 0
#
# # the get_box_iou is not perfect, let say we have 2 truth boxes, one with
# # (0, 0, 4, 4) and one with (0, 2, 4, 4). If the predict box is
# # (0, 1, 4, 4), then iou will higher for pair (0, 0, 4, 4) compare to
# # (0, 2, 4, 4). The math make sense, but in reality, is this the result that
# # we want?
#
# print(get_box_iou([0, 0, 4, 4], [0, 1, 4, 4]))
# print(get_box_iou([0, 2, 4, 4], [0, 1, 4, 4]))

# import json
# with open("../output/metrics_out.json", "w") as outfile:
#     json.dump(
#         preds_choose_truths_map(
#             t_labels=["car", "person"],
#             t_boxes=[[0, 0, 1, 1], [1, 1, 2, 2]],
#             p_labels=["bike", "person", "car"],
#             p_boxes=[[2.1, 2.1, 3, 3], [1, 1, 1.9, 1.9], [0, 0, 0.9, 0.9]],
#             threshold=0.5,
#         ),
#         outfile
#     )
