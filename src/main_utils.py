import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from nuimages import NuImages
from torchvision.io import read_image
from torchvision.models import detection as torch_model

import metrics as m
import mrcnn_utils as mru
import yolo_utils as ylu

NUIM_DATAROOT = "../input/nuImage"
MRCNN_WEIGHTS = torch_model.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
PROCESSOR = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialization():
    nuim_obj = NuImages(
        dataroot=NUIM_DATAROOT,
        version="v1.0-val",
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

    yolo_model = (
        torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model="yolov5x",
            pretrained=True,
            verbose=False,
        )
        .to(PROCESSOR)
        .eval()
    )

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


def get_lic_metrics(lic_cmatrix):
    lic_metrics = dict()

    for label in lic_cmatrix.keys():
        if label not in lic_metrics.keys():
            lic_metrics[label] = dict()

        for t_iou in lic_cmatrix[label].keys():
            if t_iou not in lic_metrics[label].keys():
                lic_metrics[label][t_iou] = dict()

            precisions = list()
            recalls = list()

            highest_f1 = 0
            highest_f1_at_conf = None

            for t_conf in lic_cmatrix[label][t_iou].keys():
                cmatrix = lic_cmatrix[label][t_iou][t_conf]
                aprf1 = m.calc_aprf1(cmatrix)

                lic_metrics[label][t_iou][t_conf] = aprf1

                # prep for AP metric and highest F1
                precisions.append(aprf1["precision"])
                recalls.append(aprf1["recall"])

                if highest_f1 <= aprf1["f1"]:
                    highest_f1 = aprf1["f1"]
                    highest_f1_at_conf = t_conf

            target_dict = lic_metrics[label][t_iou]
            target_dict["highest_f1"] = highest_f1
            target_dict["highest_f1_at_conf"] = highest_f1_at_conf

            # print(precisions)
            # print(recalls)

            target_dict["AP11"] = m.calc_ap(
                precisions=precisions, recalls=recalls, interpolation_num=11
            )
            target_dict["AP101"] = m.calc_ap(
                precisions=precisions, recalls=recalls, interpolation_num=101
            )

    return lic_metrics


def get_ap101_at_iou(lic_metrics, t_iou):
    class_ap101 = dict()

    for label in lic_metrics.keys():
        if t_iou not in lic_metrics[label].keys():
            raise ValueError(
                f"No AP101 measured at t_iou={t_iou} for {label} class"
            )

        class_ap101[label] = lic_metrics[label][t_iou]["AP101"]

    return class_ap101


def get_i_mean_ap(lic_metrics):
    supported_labels = list(lic_metrics.keys())
    supported_ious = sorted(list(lic_metrics[supported_labels[0]].keys()))

    mean_ap11s, mean_ap101s = list(), list()

    for t_iou in supported_ious:
        sum_ap11, sum_ap101 = 0, 0

        for label in supported_labels:
            sum_ap11 += lic_metrics[label][t_iou]["AP11"]
            sum_ap101 += lic_metrics[label][t_iou]["AP101"]

        mean_ap11s.append(sum_ap11 / len(supported_labels))
        mean_ap101s.append(sum_ap101 / len(supported_labels))

    return {
        "t_ious": supported_ious,
        "mAP11s": mean_ap11s,
        "mAP101s": mean_ap101s,
    }


def output_ap101_barchart_at_iou(
        model1_lic_metrics, model2_lic_metrics, t_iou
):
    model1_ap101_at_iou = get_ap101_at_iou(model1_lic_metrics, t_iou)
    model2_ap101_at_iou = get_ap101_at_iou(model2_lic_metrics, t_iou)

    labels, m1_aps, m2_aps = list(), list(), list()
    for label in sorted(model1_ap101_at_iou.keys()):
        labels.append(label)
        m1_aps.append(model1_ap101_at_iou[label])
        m2_aps.append(model2_ap101_at_iou[label])

    plt.subplots()

    title = f'AP101 of Mask R-CNN and YOLOv5 per class at t_iou = {t_iou}'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('IoU Threshold', fontsize=14)
    plt.ylabel('AP101 Score', fontsize=14)

    index = np.arange(len(labels))
    bar_width = 0.2
    opacity = 0.8

    plt.bar(
        index, m1_aps, bar_width,
        alpha=opacity, color='m', label='Mask R-CNN'
    )

    plt.bar(
        index + bar_width, m2_aps, bar_width,
        alpha=opacity, color='g', label='YOLOv5'
    )

    plt.ylim(0, 1)
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    plt.savefig(
        f"../output/Mask_R-CNN_vs_YOLOv5_AP101_at_iou{t_iou}.png", dpi=300
    )


def output_linechart_w_dataframe(
        pandas_dataframe,
        title="",
        xlabel="",
        ylabel="",
        l1style="-Dr",
        l2style="-vc",
        outfile_name=None,
):
    df = pandas_dataframe
    df_keys = pandas_dataframe.keys()

    if not outfile_name:
        outfile_name = "_".join(df.keys)

    df.to_csv(
        f"../output/metadata/{outfile_name}.csv", encoding="utf-8", index=False
    )

    plt.figure(random.randint(len(title) + 100, len(title) + 100 * 2))

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    max_y_value = max([max(df[df_keys[1]]), max(df[df_keys[2]])])
    for _val in range(0, 100, 25):
        if _val / 100 > max_y_value:
            plt.ylim(0, _val / 100)
            break

    plt.xlim(min(df[df_keys[0]]) - 0.05, max(df[df_keys[0]]) + 0.05)
    plt.grid(True)

    plt.plot(df[df_keys[0]], df[df_keys[1]], l1style, label=df_keys[1])
    plt.plot(df[df_keys[0]], df[df_keys[2]], l2style, label=df_keys[2])
    plt.legend(loc="upper right")

    plt.savefig(f"../output/{outfile_name}.png", dpi=300)
