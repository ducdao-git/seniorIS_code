# import sklearn.metrics as skm
#
#
# def calc_f1(precision, recall):
#     numerator = 2 * precision * recall
#     denominator = precision + recall
#
#     if numerator == 0:
#         return 0
#
#     return numerator / denominator
#
#
# def calc_accuracy(confusion_matrix):  # list[list[int]]
#     """
#     calculate the accuracy value for the given confusion matrix
#     :param confusion_matrix: assume to have format [[<TN>, <FP>], [<FN>, <TP>]]
#     :return: <float> the accuracy value
#     """
#     true_negative, _ = confusion_matrix[0]
#     _, true_positive = confusion_matrix[1]
#
#     numerator = true_positive + true_negative
#     denominator = sum(confusion_matrix[0]) + sum(confusion_matrix[1])
#
#     if numerator == 0:
#         return 0
#
#     return numerator / denominator
#
#
# def calc_precision(confusion_matrix):  # list[list[int]]
#     """
#     calculate the precision value for the given confusion matrix
#     :param confusion_matrix: assume to have format [[<TN>, <FP>], [<FN>, <TP>]]
#     :return: <float> the precision value
#     """
#     _, false_positive = confusion_matrix[0]
#     _, true_positive = confusion_matrix[1]
#
#     numerator = true_positive
#     denominator = true_positive + false_positive
#
#     if numerator == 0:
#         return 0
#
#     return numerator / denominator
#
#
# def calc_recall(confusion_matrix):
#     """
#     calculate the recall value for the given confusion matrix
#     :param confusion_matrix: assume to have format [[<TN>, <FP>], [<FN>, <TP>]]
#     :return: <float> the recall value
#     """
#     false_negative, true_positive = confusion_matrix[1]
#
#     numerator = true_positive
#     denominator = true_positive + false_negative
#
#     if numerator == 0:
#         return 0
#
#     return numerator / denominator
#
#
# def multilabel_th_cmatrix(pred_truth_map, thresholds):
#     """
#     generate confusion matrix for multilabel result truth_pred at each
#     threshold
#
#     :param pred_truth_map:
#     :param thresholds:
#     :return:
#     """
#     result_by_threshold = dict()
#     for threshold in thresholds:
#         result_by_threshold[threshold] = dict()
#         truth_labels = list()
#         pred_labels = list()
#
#         for item in pred_truth_map:
#             if item["iou_score"] and item["iou_score"] >= threshold:
#                 truth_labels.append(item["truth_obj"].t_label)
#                 pred_labels.append(item["pred_obj"].p_label)
#
#             elif item["iou_score"]:
#                 # when the model predict a label for object at this location
#                 # but the iou_score is low, should we just add the
#                 # truth_label to the list and have the pred_label to be
#                 # background. Or we should add another case where
#                 # truth_label is background and pred_label be added to the
#                 # pred_labels list. If we do both, then for every truth_pred
#                 # pair, that have low iou score, will be considered as 2
#                 # errors in our evaluation
#
#                 truth_labels.append(item["truth_obj"].t_label)
#                 pred_labels.append("background")
#
#                 truth_labels.append("background")
#                 pred_labels.append(item["pred_obj"].p_label)
#
#             elif item["truth_obj"]:
#                 truth_labels.append(item["truth_obj"].t_label)
#                 pred_labels.append("background")
#
#             else:
#                 truth_labels.append("background")
#                 pred_labels.append(item["pred_obj"].p_label)
#
#         # print(f"\n{truth_labels}\n{pred_labels}")
#
#         unique_labels = list(set(truth_labels).union(set(pred_labels)))
#         multi_cmatrix = skm.multilabel_confusion_matrix(
#             y_true=truth_labels,
#             y_pred=pred_labels,
#             labels=unique_labels,
#         ).astype(int)
#
#         for i in range(len(unique_labels)):
#             result_by_threshold[threshold][unique_labels[i]] = multi_cmatrix[i]
#
#     return result_by_threshold
#
#
# def preds_choose_truths_map(truth_objs, pred_objs, threshold=0.5):
#     """
#     This greedy method take a list of predictions assuming the predictions
#     are sort from high confident to low confident in the list. For each
#     prediction, the algorithm find the best fit truth box and store the
#     t_box and p_box in a dict. It also made the box map to background if they
#     not fit above the threshold.
#
#     return: list of dict. The return list have the same order as the p_boxes
#     arg (from prediction with the highest confident to lowest). The return
#     dict in the list has form:
#     {
#         iou_score: <float> repr intersection over union of p_box and t_box
#         pred_obj: <PredictClass> repr a pred object -- has the highest iou
#             with the truth_obj compare to other truth obj
#         truth_obj: <TruthClass> repr a truth object -- has the highest iou
#             with the pred_obj compare to other pred obj
#     }
#     """
#     if threshold <= 0 or threshold > 1:
#         raise ValueError(f"threshold must in (0, 1]")
#
#     pred_truth_index_map = dict()
#     pred_index_iou_map = dict()
#     for i in range(len(pred_objs)):
#         highest_iou = 0
#         highest_truth_index = None
#
#         for j in range(len(truth_objs)):
#             iou_score = get_box_iou(
#                 t_box_cord=truth_objs[j].t_bbox,
#                 p_box_cord=pred_objs[i].p_bbox,
#             )
#
#             if iou_score > highest_iou:
#                 if j in pred_truth_index_map.values():
#                     continue
#
#                 highest_iou = iou_score
#                 highest_truth_index = j
#
#         if highest_iou > threshold:
#             pred_truth_index_map[i] = highest_truth_index
#             pred_index_iou_map[i] = highest_iou
#
#     pred_truth_map = list()
#     for i in range(len(pred_objs)):
#         if i in pred_truth_index_map.keys():
#             pred_truth_map.append(
#                 {
#                     "iou_score": pred_index_iou_map[i],
#                     "pred_obj": pred_objs[i],
#                     "truth_obj": truth_objs[pred_truth_index_map[i]],
#                 }
#             )
#         else:
#             pred_truth_map.append(
#                 {
#                     "iou_score": None,
#                     "pred_obj": pred_objs[i],
#                     "truth_obj": None,
#                 }
#             )
#
#     for j in range(len(truth_objs)):
#         if j not in pred_truth_index_map.values():
#             pred_truth_map.append(
#                 {
#                     "iou_score": None,
#                     "pred_obj": None,
#                     "truth_obj": truth_objs[j],
#                 }
#             )
#
#     return pred_truth_map
#
#
# def get_box_iou(t_box_cord, p_box_cord):
#     p_box = {
#         "x1": p_box_cord[0],
#         "y1": p_box_cord[1],
#         "x2": p_box_cord[2],
#         "y2": p_box_cord[3],
#     }
#
#     t_box = {
#         "x1": t_box_cord[0],
#         "y1": t_box_cord[1],
#         "x2": t_box_cord[2],
#         "y2": t_box_cord[3],
#     }
#
#     if p_box["x1"] > p_box["x2"] or p_box["y1"] > p_box["y2"]:
#         raise ValueError("p_box_cord must in (xmin, ymin, xmax, ymax) format")
#
#     if t_box["x1"] > t_box["x2"] or t_box["y1"] > t_box["y2"]:
#         raise ValueError("t_box_cord must in (xmin, ymin, xmax, ymax) format")
#
#     # coordinate of the intersection rectangle
#     inter_x1 = max(p_box["x1"], t_box["x1"])
#     inter_y1 = max(p_box["y1"], t_box["y1"])
#     inter_x2 = min(p_box["x2"], t_box["x2"])
#     inter_y2 = min(p_box["y2"], t_box["y2"])
#
#     if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
#         return 0
#
#     # area of the intersection rectangle
#     intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
#
#     # area of predicted box and truth box
#     p_box_area = (p_box["x2"] - p_box["x1"]) * (p_box["y2"] - p_box["y1"])
#     t_box_area = (t_box["x2"] - t_box["x1"]) * (t_box["y2"] - t_box["y1"])
#
#     return intersection_area / (p_box_area + t_box_area - intersection_area)
