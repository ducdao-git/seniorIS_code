from pprint import pprint

import torch
from nuimages import NuImages
from torchvision.io import read_image
from torchvision.models import detection as torch_model
import numpy as np
from label_mapping import supported_label
import metrics as m
import mrcnn_utils as mru
import yolo_utils as ylu

NUIM_DATAROOT = "../input/nuImage"
MRCNN_WEIGHTS = torch_model.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
PROCESSOR = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialization():
    nuim_obj = NuImages(
        dataroot=NUIM_DATAROOT,
        version="v1.0-mini",
        verbose=False,
        lazy=True,
    )
    print(type(nuim_obj.sample), len(nuim_obj.sample))

    mrcnn_model = (
        torch_model.maskrcnn_resnet50_fpn(
            weights=MRCNN_WEIGHTS,
            progress=True,
            num_classes=91,
        )
        .to(PROCESSOR)
        .eval()
    )

    # yolo_model = torch.hub.load(
    #     repo_or_dir="ultralytics/yolov5",
    #     model="yolov5x",
    #     pretrained=True,
    #     verbose=False,
    # ).to(PROCESSOR).eval()
    yolo_model = None  # to be removed

    return nuim_obj, mrcnn_model, yolo_model


def get_mrcnn_predict_objs(mrcnn_model, img_path):
    img_tensor = read_image(img_path)
    transform_img_tensor = MRCNN_WEIGHTS.transforms()(img_tensor)

    p_labels, p_scores, p_bboxes, p_masks = mru.get_mrcnn_outputs(
        mrcnn_model=mrcnn_model,
        mrcnn_weights=MRCNN_WEIGHTS,
        img_tensor=transform_img_tensor.to(PROCESSOR),
        supported_label_only=True,
    )

    pred_objs = mru.mrcnn_output_to_objs(p_labels, p_scores, p_bboxes, p_masks)

    return pred_objs


def get_yolo_predict_objs(yolo_model, img_path):
    pred_objs = ylu.get_yolo_predict_objs(
        yolo_model=yolo_model,
        img_path=img_path,
        supported_label_only=True,
    )

    return pred_objs


def get_pred_truth_map(truth_objs, pred_objs):
    pt_map = m.preds_choose_truths_map(truth_objs, pred_objs)

    pt_lmap = {
        "iou_scores": list(),
        "confidence_scores": list(),
        "pred_labels": list(),
        "truth_labels": list(),
    }

    for mapping in pt_map:
        p_score, p_label, t_label = None, None, None

        if mapping["pred_obj"]:
            p_score = mapping["pred_obj"].p_score
            p_label = mapping["pred_obj"].p_label

        if mapping["truth_obj"]:
            t_label = mapping["truth_obj"].t_label

        pt_lmap["iou_scores"].append(mapping["iou_score"])
        pt_lmap["confidence_scores"].append(p_score)
        pt_lmap["pred_labels"].append(p_label)
        pt_lmap["truth_labels"].append(t_label)

    # for label in pt_lmap.keys():
    #     print(f'{label} ({len(pt_lmap[label])}): {pt_lmap[label]}')
    return pt_map, pt_lmap


def init_lic_cmatrix(supported_labels, t_ious, t_confidences):
    lic = dict()

    for label in supported_labels:
        if label not in lic.keys():
            lic[label] = dict()

        for t_iou in t_ious:
            if t_iou not in lic[label].keys():
                lic[label][t_iou] = dict()

            for t_conf in t_confidences:
                if t_conf not in lic[label][t_iou].keys():
                    lic[label][t_iou][t_conf] = np.array([[0, 0], [0, 0]])

    return lic


def update_lic_cmatrix(
    updating_lic, pred_truth_label_map, t_confidences, t_ious
):
    for t_iou in t_ious:
        for t_conf in t_confidences:
            ml_cmatrix = m.multilabel_cmatrix(
                pred_truth_label_map=pred_truth_label_map,
                t_confidence=t_conf,
                t_iou=t_iou,
            )

            for label in ml_cmatrix.keys():
                updating_lic[label][t_iou][t_conf] += ml_cmatrix[label]

    print(updating_lic)
