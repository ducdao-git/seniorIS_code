import json
import time
from pprint import pprint

import torch
from nuimages import NuImages

import metrics as m
from label_mapping import label_int_map
from mrcnn_utils import get_mrcnn_predict_objs
from nuimg_sample import NuImgSample
from truth_class import TruthClass
from yolo_utils import get_yolov5_predict_objs

# torch.set_printoptions(profile="full")
# torch.set_printoptions(linewidth=10000)

CONFIDENT_THRESHOLD = 0.5
MAP_THRESHOLD = 0.5  # y_pred and y_truth must have higher IOU to be mapped tgt
IOU_THRESHOLDS = [t / 100 for t in range(50, 95, 5)]
IMG_INDEX = range(0, 50)
SUPPORTED_LABELS = label_int_map

nuImages_dataroot = "../input/nuImage"
nuim_obj = NuImages(
    dataroot=nuImages_dataroot,
    version="v1.0-mini",
    verbose=False,
    lazy=True,
)

computation_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(computation_device)


def get_truth_objs(img_index):
    # ------------ get the image ------------ #
    sample_img = nuim_obj.sample[img_index]
    sample_img = NuImgSample(
        nuimg=nuim_obj,
        sample_token=sample_img["token"],
        nuim_mrcnn_label_only=True,
    )

    sample_img_path = sample_img.get_sample_img_path()
    # sample_img_int = read_image(sample_img_path)

    # ----------------- generate ground truth ----------------- #
    t_bboxes = sample_img.get_objs_bbox()
    t_labels = sample_img.get_objs_mrcnn_category()
    t_masks = sample_img.get_objs_mask()

    truth_objs = list()
    for i in range(len(t_labels)):
        truth_obj = TruthClass(
            label=t_labels[i],
            bbox=t_bboxes[i],
            mask=t_masks[i],
        )
        truth_objs.append(truth_obj)

    return sample_img_path, truth_objs


def update_batch_th_cmatrix(batch_th_cmatrix, model_name, th_cmatrix):
    if model_name not in batch_th_cmatrix.keys():
        batch_th_cmatrix[model_name] = dict()

        for t in IOU_THRESHOLDS:
            batch_th_cmatrix[model_name][t] = dict()

    for th in th_cmatrix.keys():
        target_th = batch_th_cmatrix[model_name][th]
        source_th = th_cmatrix[th]

        for label in source_th.keys():
            if label in target_th:
                target_th[label] += source_th[label]
            else:
                target_th[label] = source_th[label]

    return batch_th_cmatrix


def get_batch_label_eval(batch_th_cmatrix):
    batch_label_eval = dict()

    # -------------------------------------------------------------------------
    # calculate accuracy, precision, recall at each threshold for each label
    for model in batch_th_cmatrix:
        for th in batch_th_cmatrix[model]:
            for label in batch_th_cmatrix[model][th]:
                metric_dict = dict()
                confusion_matrix = batch_th_cmatrix[model][th][label]

                metric_dict["accuracy"] = m.calc_accuracy(confusion_matrix)
                metric_dict["precision"] = m.calc_precision(confusion_matrix)
                metric_dict["recall"] = m.calc_recall(confusion_matrix)

                metric_dict["f1"] = m.calc_f1(
                    precision=metric_dict["precision"],
                    recall=metric_dict["recall"],
                )

                if model not in batch_label_eval:
                    batch_label_eval[model] = dict()
                    for _l in SUPPORTED_LABELS.keys():
                        batch_label_eval[model][_l] = dict()

                batch_label_eval[model][label][th] = metric_dict

    # -------------------------------------------------------------------------
    # calculate average precision (AP) across all threshold for each label
    for model in batch_label_eval:
        target_loc = batch_label_eval[model]

        for label in target_loc:
            precisions = list()
            recalls = list()

            label_dict = target_loc[label]
            for th in label_dict:
                precisions.append(label_dict[th]["precision"])
                recalls.append(label_dict[th]["recall"])

            precisions.append(1)
            recalls.append(0)

            label_dict["AP"] = 0
            for k in range(len(precisions) - 1):
                k_eval = (recalls[k] - recalls[k + 1]) * precisions[k]
                label_dict["AP"] += k_eval

        target_loc["mAP"] = sum(
            [target_loc[label]["AP"] for label in target_loc]
        ) / len(SUPPORTED_LABELS)

    return batch_label_eval


def main():
    start_time = time.time()
    # -------------------------------------------------------------------------
    # create dict to store confusion matrix and evaluation metrics
    batch_th_cmatrix = dict()  # batch, threshold, multilabel confusion matrix

    # -------------------------------------------------------------------------
    # generate confusion matrix for each label at different threshold
    for img_index in IMG_INDEX:
        print(f"\n{'-' * 9} img_index: {img_index} {'-' * 9}")
        sample_img_path, truth_objs = get_truth_objs(img_index)

        # confusion matrix for mask r-cnn
        mrcnn_pred_objs = get_mrcnn_predict_objs(
            computation_device=computation_device,
            sample_img_path=sample_img_path,
            thresh_hold=CONFIDENT_THRESHOLD,
        )
        mrcnn_pred_truth_map = m.preds_choose_truths_map(
            truth_objs=truth_objs,
            pred_objs=mrcnn_pred_objs,
            threshold=MAP_THRESHOLD,
        )
        update_batch_th_cmatrix(
            batch_th_cmatrix=batch_th_cmatrix,
            model_name="mrcnn",
            th_cmatrix=m.multilabel_th_cmatrix(
                pred_truth_map=mrcnn_pred_truth_map,
                thresholds=IOU_THRESHOLDS,
            ),
        )

        # confusion matrix for YOLOv5
        yolo_pred_objs = get_yolov5_predict_objs(
            computation_device, sample_img_path
        )
        yolo_pred_truth_map = m.preds_choose_truths_map(
            truth_objs=truth_objs,
            pred_objs=yolo_pred_objs,
            threshold=MAP_THRESHOLD,
        )
        update_batch_th_cmatrix(
            batch_th_cmatrix=batch_th_cmatrix,
            model_name="yolov5",
            th_cmatrix=m.multilabel_th_cmatrix(
                pred_truth_map=yolo_pred_truth_map,
                thresholds=IOU_THRESHOLDS,
            ),
        )

    # -------------------------------------------------------------------------
    # print out or dump to json with the variable name
    print(f"\n{'-' * 9} confusion matrix, multilabel, multi-threshold")
    pprint(batch_th_cmatrix)

    batch_label_eval = get_batch_label_eval(batch_th_cmatrix)
    with open("../output/batch_label_eval.json", "w") as outfile:
        json.dump(batch_label_eval, outfile)

    print(f"runtime: {time.time() - start_time}")


main()
