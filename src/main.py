import matplotlib.pyplot as plt
import numpy as np
import time
import json
import torch
from nuimages import NuImages
from torchvision.io import read_image
from torchvision.transforms import functional as torch_func
from torchvision.utils import draw_bounding_boxes

import metrics as m
from mrcnn_utils import get_mask_rcnn, get_mrcnn_outputs
from nuimg_sample import NuImgSample

from truth_class import TruthClass
from predict_class import PredictClass

from pprint import pprint

# torch.set_printoptions(profile="full")
# torch.set_printoptions(linewidth=10000)

CONFIDENT_THRESHOLD = 0.5
MAP_THRESHOLD = 0.5  # y_pred and y_truth must have higher IOU to be mapped tgt
IOU_THRESHOLDS = [t / 100 for t in range(50, 95, 5)]

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


def get_truth_objs(img_index):
    # ------------ get the image ------------ #
    sample_img = nuim_obj.sample[img_index]
    sample_img = NuImgSample(
        nuimg=nuim_obj,
        sample_token=sample_img["token"],
        nuim_mrcnn_label_only=True,
    )

    sample_img_path = sample_img.get_sample_img_path()
    sample_img_int = read_image(sample_img_path)

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

    return sample_img_int, truth_objs


def get_mrcnn_predict_objs(sample_img_int, thresh_hold=CONFIDENT_THRESHOLD):
    # ----------------- generate predicted ----------------- #
    mrcnn_model, mrcnn_weights = get_mask_rcnn(computation_device)
    transform_sample_img_int = mrcnn_weights.transforms()(sample_img_int)

    p_scores, p_labels, p_bboxes, p_masks = get_mrcnn_outputs(
        mrcnn_model=mrcnn_model,
        mrcnn_weights=mrcnn_weights,
        image=transform_sample_img_int,
        thresh_hold=thresh_hold,
        nuim_mrcnn_label_only=True,
    )

    pred_objs = list()
    for i in range(len(p_labels)):
        pred_obj = PredictClass(
            label=p_labels[i],
            score=p_scores[i],
            bbox=p_bboxes[i],
            mask=p_masks[i],
        )
        pred_objs.append(pred_obj)

    return transform_sample_img_int, pred_objs


def update_batch_th_cmatrix(batch_th_cmatrix, th_cmatrix):
    for th in th_cmatrix.keys():
        target_th = batch_th_cmatrix[th]
        source_th = th_cmatrix[th]

        for label in source_th.keys():
            if label in target_th:
                target_th[label] += source_th[label]
            else:
                target_th[label] = source_th[label]

    return batch_th_cmatrix


def get_batch_th_eval(batch_th_cmatrix):
    batch_th_eval = dict()
    for th in batch_th_cmatrix.keys():
        batch_th_eval[th] = dict()

    for th in batch_th_cmatrix:
        for label in batch_th_cmatrix[th]:
            metric_dict = dict()
            confusion_matrix = batch_th_cmatrix[th][label]

            metric_dict["accuracy"] = m.calc_accuracy(confusion_matrix)
            metric_dict["precision"] = m.calc_precision(confusion_matrix)
            metric_dict["recall"] = m.calc_recall(confusion_matrix)

            metric_dict["f1"] = m.calc_f1(
                precision=metric_dict["precision"],
                recall=metric_dict["recall"],
            )

            batch_th_eval[th][label] = metric_dict

    return batch_th_eval


def main():
    start_time = time.time()
    # -------------------------------------------------------------------------
    # create dict to store confusion matrix and evaluation metrics
    batch_th_cmatrix = dict()  # batch, threshold, multilabel confusion matrix
    for t in IOU_THRESHOLDS:
        batch_th_cmatrix[t] = dict()

    # -------------------------------------------------------------------------
    # generate confusion matrix for each label at different threshold
    for img_index in range(0, 50):
        print(f"\n{'-' * 9} img_index: {img_index} {'-' * 9}")

        sample_img_int, truth_objs = get_truth_objs(img_index)
        sample_img_int, pred_objs = get_mrcnn_predict_objs(sample_img_int)

        pred_truth_map = m.preds_choose_truths_map(
            truth_objs=truth_objs,
            pred_objs=pred_objs,
            threshold=MAP_THRESHOLD,
        )
        # pprint(pred_truth_map)

        update_batch_th_cmatrix(
            batch_th_cmatrix=batch_th_cmatrix,
            th_cmatrix=m.multilabel_th_cmatrix(
                pred_truth_map=pred_truth_map,
                thresholds=IOU_THRESHOLDS,
            ),
        )

    # -------------------------------------------------------------------------
    # print out or dump to json with the variable name
    print(f"\n{'-' * 9} confusion matrix, multilabel, multi-threshold")
    pprint(batch_th_cmatrix)

    batch_th_eval = get_batch_th_eval(batch_th_cmatrix)
    with open("../output/batch_th_eval.json", "w") as outfile:
        json.dump(batch_th_eval, outfile)

    print(f"runtime: {time.time() - start_time}")


main()
