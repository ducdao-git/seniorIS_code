import os.path
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

NUIM_DATAROOT = "../input/nuImages"
MRCNN_WEIGHTS = torch_model.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
# PROCESSOR = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSOR = torch.device("cpu")


def initialization():
    """
    program initialization that generate the nuImages object (to asset
    images dataset), the mask r-cnn and yolo model

    :return: nuImages object, mask r-cnn model object, yolov5 model object
    """
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
            # coco define 91 class label but only use 80 in the dataset
        )
        .to(PROCESSOR)
        .eval()
    )

    yolo_model = (
        torch.hub.load(
            repo_or_dir="ultralytics/yolov5",
            model="yolov5x",
            pretrained=True,  # default to use COCO pretrained weight
            verbose=False,
        )
        .to(PROCESSOR)
        .eval()
    )

    return nuim_obj, mrcnn_model, yolo_model


def get_mrcnn_predict_objs(mrcnn_model, img_path):
    """
    generate object prediction using the mask r-cnn model for an image

    :param mrcnn_model: the mask r-cnn model object
    :param img_path: the path to the image to generate prediction for

    :return: list of PredictObject (defined in predict_object.py)
    """
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
    """
    generate object prediction using the yolo version 5 model for an image

    :param yolo_model: the yolov5 model object
    :param img_path: the path to the image to generate prediction for

    :return: list of PredictObject (defined in predict_object.py)
    """
    pred_objs = ylu.get_yolo_predict_objs(
        yolo_model=yolo_model,
        img_path=img_path,
        supported_label_only=True,
    )

    return pred_objs


def get_pred_truth_map(truth_objs, pred_objs):
    """
    mapping the TruthObject list with the PredictObject list

    :param truth_objs: list of TruthObject (defined in truth_object.py)
    :param pred_objs: list of PredictObject (defined in predict_object.py)

    :return: 2 lists of dictionary: pt_map and pt_lmap
    - pt_map: a list of dict where each dict has 3 fields -- one for a
        TruthObject, one for a PredictObject, and one for the IoU score between
        the predicted and truth object
    - pt_lmap: a list of dict where each dict has 3 fields -- one for a
        TruthObject's label, one for a PredictObject's label, one for
        PredictObject's confidence score, and one for the IoU score between the
        predicted and truth object
    """
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
    """
    initialize an empty nested dictionary to store the confusion matrix in.
    the nested dict will have the following format:
    {
        <label> : {
            <IoU_threshold> : {
                <confidence_threshold>: {
                    <a 2D numpy array repr confusion matrix with for this
                    label at this particular IoU and confidence threshold>
                },
                ...,
            },
            ...,
        },
        ..,
    }

    :param supported_labels: list of str, each str repr a supported label
    :param t_ious: list of float, each float repr an IoU threshold
    :param t_confidences: list of float, each float repr a confidence threshold

    :return: return the nested dictionary (described previously)
    """
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
    """
    updating a lic_cmatric using a list of dictionary where each
    dict contain the ground-truth label, predicted label, predicted
    confidence score, and IoU score. The confusion matrix for each label at
    each pair of IoU and confidence threshold is generated using the
    `multilabel_cmatrix` function defined in metrics.py

    :param updating_lic: the lic_cmatrix to be updated. the lic_cmatrix is a
        nested dict that is first initialized with the `init_lic_cmatrix`
        function
    :param pred_truth_label_map: list of dictionary where each
        dict contain the ground-truth label, predicted label, predicted
        confidence score, and IoU score.
    :param t_confidences: list of confidence threshold in which the function
        update the confusion matrix for
    :param t_ious: list of IoU threshold in which the function update the
        confusion matrix for

    :return: None
    """
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
    """
    helper function that call function in metrics.py to calculate different
    metrics used to evaluate the model for each label at different
    combination of IoU and confidence threshold. These metrics are: accuracy,
    precision, recall, f-score (f1-score), 11-point and 101-point interpolation
    mAP. These calculated metric is then stored in a nested dictionary with
    the following format:
    {
        <label>: {
            <IoU_threshold>: {
                <confidence_threshold_0>: {
                    "accuracy": a float,
                    "precision": a float,
                    "recall": a float,
                    "f1": a float,
                },
                ...,
                <confidence_threshold_n>: {
                    "accuracy": a float,
                    "precision": a float,
                    "recall": a float,
                    "f1": a float,
                },
                "highest_f1": a float,
                "highest_f1_at_conf": a float repr a confidence threshold,
                "AP11": a float,
                "AP101": a float,
            },
            ...,
        },
        ...,
    }

    :param lic_cmatrix: the nested dictionary (keys level: label -> IoU
        threshold -> confidence threshold) used to store the confusion matrix (
        a 2D list)

    :return: the nested dict that stored evaluated metrics (described
    previously)
    """
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
    """
    get a list of 101-point interpolation AP value of different classes at
    an IoU threshold value

    :param lic_metrics: the nested dict that hold all the computed metrics,
        generated by function `get_lic_metrics`
    :param t_iou: a float repr an IoU threshold

    :return: a list of 101-point interpolation AP value of different classes at
    an IoU threshold value
    """
    class_ap101 = dict()

    for label in lic_metrics.keys():
        if t_iou not in lic_metrics[label].keys():
            raise ValueError(
                f"No AP101 measured at t_iou={t_iou} for {label} class"
            )

        class_ap101[label] = lic_metrics[label][t_iou]["AP101"]

    return class_ap101


def get_i_mean_ap(lic_metrics):
    """
    get mean Average Precision (mAP) of the model at each different IoU
    threshold

    :param lic_metrics: the nested dict that hold all the computed metrics,
        generated by the function `get_lic_metrics`

    :return: a dict with 3 fields repr list of IoU threshold, list of
    11-point, and list of 101-point interpolation mAP. The 3 lists'
    elements are correspond by position
    """
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
        mrcnn_lic_metrics, yolov5_lic_metrics, t_iou
):
    """
    generate a barchart that compare the 101-point interpolation AP value of
    Mask R-CNN and YOLO model per class at a specific IoU threshold. Can be
    adapted and generalize to comparing any 2 model, not just Mask R-CNN
    and YOLO. The generated figure is saved in the `output` directory.

    :param mrcnn_lic_metrics: the nested dict that hold all the computed
        metrics, generated by the function `get_lic_metrics` for the Mask R-CNN
        model
    :param yolov5_lic_metrics: the nested dict that hold all the computed
        metrics, generated by the function `get_lic_metrics` for the YOLOv5
        model
    :param t_iou: a float repr an IoU threshold

    :return: None
    """
    model1_ap101_at_iou = get_ap101_at_iou(mrcnn_lic_metrics, t_iou)
    model2_ap101_at_iou = get_ap101_at_iou(yolov5_lic_metrics, t_iou)

    labels, m1_aps, m2_aps = list(), list(), list()
    for label in sorted(model1_ap101_at_iou.keys()):
        labels.append(label)
        m1_aps.append(model1_ap101_at_iou[label])
        m2_aps.append(model2_ap101_at_iou[label])

    plt.subplots()

    title = f'AP101 of Mask R-CNN and YOLOv5 per class at t_iou = {t_iou}'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Class Label', fontsize=14)
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


def output_tp_barchart_at_iou_conf(
        truth_counter, mrcnn_lic_cmatrix, yolov5_lic_cmatrix, t_iou, t_conf
):
    """
    generate a barchart that compare the number of occurrence of each class
    predicted by mask r-cnn, yolov5, versus the truth occurrence at a
    specific IoU and confidence threshold. Can be adapted and generalize to
    comparing any 2 model, not just Mask R-CNN and YOLO. The generated figure
    is saved in the `output` directory.

    :param truth_counter: a dict that hold the number of ground-truth
        occurrence of each class across all processed images
    :param mrcnn_lic_cmatrix: the nested dict that hold the confusion
        matrices initialized by the function `init_lic_cmatrix` and updated by
        `update_lic_cmatrix` for the Mask R-CNN model
    :param yolov5_lic_cmatrix: the nested dict that hold the confusion
        matrices initialized by the function `init_lic_cmatrix` and updated by
        `update_lic_cmatrix` for the YOLO model
    :param t_iou: a float repr an IoU threshold
    :param t_conf: a float repr a confidence threshold

    :return: None
    """
    labels = sorted(
        set(truth_counter.keys()).union(set(mrcnn_lic_cmatrix.keys())).union(
            set(yolov5_lic_cmatrix.keys())
        )
    )

    t_counter, m1_counter, m2_counter = list(), list(), list()
    for label in sorted(labels):
        t_counter.append(
            truth_counter[label] if label in truth_counter else 0
        )

        m1_counter.append(
            mrcnn_lic_cmatrix[label][t_iou][t_conf][1][1] if
            label in mrcnn_lic_cmatrix else 0
        )

        m2_counter.append(
            yolov5_lic_cmatrix[label][t_iou][t_conf][1][1] if
            label in yolov5_lic_cmatrix else 0
        )

    plt.subplots()

    title = f'Models\'s TP vs. ground-truth per class at t_iou={t_iou}, ' \
            f't_conf={t_conf}  '

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel('True Positive Count', fontsize=14)

    index = np.arange(len(labels))
    bar_width = 0.2
    opacity = 0.8

    plt.bar(
        index, m1_counter, bar_width,
        alpha=opacity, color='m', label='Mask R-CNN'
    )

    plt.bar(
        index + bar_width, m2_counter, bar_width,
        alpha=opacity, color='g', label='YOLOv5'
    )

    plt.bar(
        index + bar_width * 2, t_counter, bar_width,
        alpha=opacity, color='tab:orange', label='Ground-truth'
    )

    plt.xticks(index + bar_width, labels)
    plt.yticks([v * 1000 for v in [5, 15, 25, 35, 45]])
    plt.legend()

    plt.savefig(
        f"../output/true_positive_at_iou{t_iou}_conf{t_conf}.png", dpi=300
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
    """
    generate a linechart for a dataframe of 3 columns. The generated chart
    is stored in the `output` directory

    :param pandas_dataframe: a dataframe of 3 columns. The first column
        value is used for x-axis value. While the second and third column is
        plotted as y-values based on x-value
    :param title: str repr the title of the figure
    :param xlabel: str repr the label of the x-axis
    :param ylabel: str repr the label of the y-axis
    :param l1style: str repr style of the first line (second column). These
        style string are defined in matplotlib line chart style
    :param l2style: str repr style of the first line (third column). These
        style string are defined in matplotlib line chart style
    :param outfile_name: str repr the name of the output file, this file
        will be put in the `output` directory

    :return: None
    """
    df = pandas_dataframe
    df_keys = pandas_dataframe.keys()

    if not os.path.exists("../output/metadata/"):
        os.mkdir("../output/metadata/")

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
