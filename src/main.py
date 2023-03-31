import json
import os
import time

import numpy as np
import pandas as pd

import main_utils as mut
import nuim_util as nuu
from label_mapping import supported_label

NUIMG_INDEX = range(0, 16445)  # 16445 images in nuImage validation set
CONFIDENCE_THRESHOLDS = [t / 100 for t in range(0, 101, 10)]  # [0:0.01:1]
IOU_THRESHOLDS = [t / 100 for t in range(50, 100, 5)]  # [0.5:0.05:0.95]


def predict_with_mrcnn(mrcnn_lic_cmatrix, mrcnn_model, img_path, truth_objs):
    """
    helper function that call util functions to predict all object present
    in an image with the mask r-cnn model (second positional argument) and
    stored it in the nested dict (first positional argument)

    :param mrcnn_lic_cmatrix: the current lic_cmatric -- a nested
        dictionary used to store confusion matrix. The dict is nested by key:
        label, then IoU threshold value, then confidence threshold.
    :param mrcnn_model: Mask R-CNN model object from pytorch
    :param img_path: str repr path to the image on the system
    :param truth_objs: list of TruthObject defined in truth_object.py

    :return: inplace updating the mrcnn_lic_cmatrix -- nested dict -- but
    also return it
    """
    mrcnn_preds = mut.get_mrcnn_predict_objs(mrcnn_model, img_path)
    # print("-" * 10, "MRCNN:", len(mrcnn_preds), "-" * 10)
    print(len(mrcnn_preds), end=", ")

    mrcnn_pt_map, mrcnn_pt_lmap = mut.get_pred_truth_map(
        truth_objs, mrcnn_preds
    )

    mut.update_lic_cmatrix(
        mrcnn_lic_cmatrix, mrcnn_pt_lmap, CONFIDENCE_THRESHOLDS, IOU_THRESHOLDS
    )

    return mrcnn_lic_cmatrix


def predict_with_yolo(yolo_lic_cmatrix, yolo_model, img_path, truth_objs):
    """
    helper function that call util functions to predict all object present
    in an image with the yolo model (second positional argument) and stored it
    in the nested dict (first positional argument)

    :param yolo_lic_cmatrix: the current lic_cmatric -- a nested
        dictionary used to store confusion matrix. The dict is nested by key:
        label, then IoU threshold value, then confidence threshold.
    :param yolo_model: YOLO model object from pytorch
    :param img_path: str repr path to the image on the system
    :param truth_objs: list of TruthObject defined in truth_object.py

    :return: inplace updating the mrcnn_lic_cmatrix -- the nested dict -- but
    also return it
    """
    yolo_preds = mut.get_yolo_predict_objs(yolo_model, img_path)
    # print("-" * 10, "YOLOv5:", len(yolo_preds), "-" * 10)
    print(len(yolo_preds))

    yolo_pt_map, yolo_pt_lmap = mut.get_pred_truth_map(truth_objs, yolo_preds)

    mut.update_lic_cmatrix(
        yolo_lic_cmatrix, yolo_pt_lmap, CONFIDENCE_THRESHOLDS, IOU_THRESHOLDS
    )

    return yolo_lic_cmatrix


def evaluate_model(model_lic_cmatrix, model_name):
    """
    helper function that use util functions to generate all metrics
    evaluation and generate a linechart graph that denote the different
    between 11-point interpolation mAP and 101-point interpolation mAP value at
    different IoU thresholds. The metrics evaluation is stored in a nested
    dictionary with keys: label, then IoU threshold, then confidence
    threshold.

    :param model_lic_cmatrix: the model lic_cmatric -- a nested
        dictionary used to store confusion matrix. The dict is nested by key:
        label, then IoU threshold value, then confidence threshold.
    :param model_name: string repr the name of the model, this name will be
        used to display in the title of the line chart graph

    :return: a dataframe and a dict
    - a dataframe with each row contain 3 values: the IoU threshold,
        the 11-point interpolation mAP value, and the 101-point interpolation
        mAP value
    - a nested dict that hold all the metrics evaluation for the
        model at a specific IoU and confidence threshold.
    """
    model_lic_metrics = mut.get_lic_metrics(model_lic_cmatrix)
    model_i_mean_ap = mut.get_i_mean_ap(model_lic_metrics)

    model_eval_df = pd.DataFrame(
        {
            "t_iou": model_i_mean_ap["t_ious"],
            "mAP11": model_i_mean_ap["mAP11s"],
            "mAP101": model_i_mean_ap["mAP101s"],
        }
    )

    outfile_prefix = "_".join(model_name.split()).lower()
    mut.output_linechart_w_dataframe(
        pandas_dataframe=model_eval_df,
        title=f"{model_name} mAP11 and mAP101 at different IoU thresholds",
        xlabel="IoU threshold",
        ylabel="mAP value",
        outfile_name=f'{outfile_prefix}_mAP_Evaluation',
    )

    with open(
            f"../output/metadata/{outfile_prefix}_lic_metrics.json", "w"
    ) as outfile:
        outfile.write(json.dumps(model_lic_metrics))

    return model_eval_df, model_lic_metrics


def store_model_lic_cmatrix(model_name, model_lic_cmatrix):
    """
    Stored the nested dict (use to hold confusion matrix based on label,
    IoU and confidence threshold) as a json file to `output/metadata/`
    directory

    :param model_name: str repr the name of the model, the name will be
        inserted to the beginning of the json file's name
    :param model_lic_cmatrix: the model lic_cmatric -- a nested dictionary
        used to store confusion matrix. The dict is nested by key: label,
        then IoU threshold value, then confidence threshold.

    :return: None
    """
    if not os.path.exists("../output/metadata/"):
        os.makedirs("../output/metadata/")

    output_lic_cmatrix = dict()
    for label in sorted(model_lic_cmatrix.keys()):
        if label not in output_lic_cmatrix.keys():
            output_lic_cmatrix[label] = dict()

        for t_iou in model_lic_cmatrix[label].keys():
            if t_iou not in output_lic_cmatrix[label].keys():
                output_lic_cmatrix[label][t_iou] = dict()

            for t_conf in model_lic_cmatrix[label][t_iou].keys():
                lic_cm_list = model_lic_cmatrix[label][t_iou][t_conf].tolist()
                output_lic_cmatrix[label][t_iou][t_conf] = lic_cm_list

    with open(
            f"../output/metadata/{model_name}_lic_cmatrix.json", "w"
    ) as outfile:
        outfile.write(json.dumps(output_lic_cmatrix))


def load_lic_cmatrix_to(load_lic_path, lic_tobe_update):
    """
    load the lic_cmatrix that stored in a json file on the system with
    `store_model_lic_cmatrix` back to the software. Please read the
    `store_model_lic_cmatrix` before use this function.

    :param load_lic_path: str repr the path to the json file that contain
        lic_cmatrix
    :param lic_tobe_update: the lic_cmatrix object that the lic_cmatrix in
        the json file will be loaded to

    :return: None
    """
    if not os.path.exists(load_lic_path):
        raise ValueError(f"load_lic_path ({load_lic_path}) not exist.")

    with open(load_lic_path, "r") as infile:
        loaded_lic = json.load(infile)

    for label in loaded_lic.keys():
        for t_iou in loaded_lic[label].keys():
            for t_conf in loaded_lic[label][t_iou].keys():
                loaded_lic_nparr = np.array(loaded_lic[label][t_iou][t_conf])

                lic_tobe_update[label][float(t_iou)][
                    float(t_conf)] += loaded_lic_nparr


def main(load_lic_cmatrix=False):
    """
    the starting point of the software, the function call util and
    helper functions to generate all confusion matrices, metrics and graph
    for the program.

    :param load_lic_cmatrix: a boolean flag that indicate if the software
        will try to load a lic_cmatrix json file into the program. If the
        flag is True, then the program assume there are 2 files with name
        format <model_name>_lic_cmatrix.json and the file
        truth_class_counter.json in `output/metadata/` directory. More on
        how these files are stored in there in the first place can be seen
        in the `stored models' confusion matrix and counter` section inside
        this main function

    :return: None
    """
    start_time = time.time()

    # ----------------------------- initialization ----------------------------
    nuim_obj, mrcnn_model, yolo_model = mut.initialization()
    truth_class_counter = dict()

    # lic_cmatrix -- a dict store the confusion matrix based on label, iou
    # threshold, and confidence threshold
    mrcnn_lic_cmatrix = mut.init_lic_cmatrix(
        supported_labels=supported_label,
        t_ious=IOU_THRESHOLDS,
        t_confidences=CONFIDENCE_THRESHOLDS,
    )

    yolo_lic_cmatrix = mut.init_lic_cmatrix(
        supported_labels=supported_label,
        t_ious=IOU_THRESHOLDS,
        t_confidences=CONFIDENCE_THRESHOLDS,
    )

    if load_lic_cmatrix:
        base_path = "../output/metadata"

        load_lic_cmatrix_to(
            f"{base_path}/mask_r-cnn_lic_cmatrix.json", mrcnn_lic_cmatrix
        )
        load_lic_cmatrix_to(
            f"{base_path}/yolov5_lic_cmatrix.json", yolo_lic_cmatrix
        )

        with open(f"{base_path}/truth_class_counter.json", "r") as inf:
            truth_class_counter = json.load(inf)

    # ------------------- generate models' confusion matrix -------------------
    for img_index in NUIMG_INDEX:
        # print("-" * 20, "nuimg_index:", img_index, "-" * 20)
        print("nuimg_index:", img_index, end=", ")

        img_path, truth_objs = nuu.get_truth_objs(
            nuim_obj, img_index, truth_class_counter
        )
        print(len(truth_objs), end=", ")

        mrcnn_lic_cmatrix = predict_with_mrcnn(
            mrcnn_lic_cmatrix, mrcnn_model, img_path, truth_objs
        )

        yolo_lic_cmatrix = predict_with_yolo(
            yolo_lic_cmatrix, yolo_model, img_path, truth_objs
        )

    print(f"Prediction time: {time.time() - start_time}")

    # -------------- stored models' confusion matrix and counter --------------
    store_model_lic_cmatrix("mask_r-cnn", mrcnn_lic_cmatrix)
    store_model_lic_cmatrix("yolov5", yolo_lic_cmatrix)

    with open(f"../output/metadata/truth_class_counter.json", "w") as outf:
        outf.write(json.dumps(truth_class_counter))

    # ---------- evaluate models' predictions using confusion matrix ----------
    mrcnn_eval_df, mrcnn_lic_metrics = evaluate_model(
        mrcnn_lic_cmatrix, model_name="Mask R-CNN"
    )
    yolo_eval_df, yolo_lic_metrics = evaluate_model(
        yolo_lic_cmatrix, model_name="YOLOv5"
    )

    # ---------------- display models' predictions evaluation -----------------
    # display line chart mAP101 at different IoU thresholds
    compare_model_mean_ap101_df = pd.DataFrame({
        "t_iou": mrcnn_eval_df["t_iou"],
        "Mask R-CNN": mrcnn_eval_df["mAP101"],
        "YOLOv5": yolo_eval_df["mAP101"],
    })
    mut.output_linechart_w_dataframe(
        pandas_dataframe=compare_model_mean_ap101_df,
        title="mAP101 of Mask R-CNN and YOLOv5 at different IoU thresholds  ",
        xlabel="IoU threshold",
        ylabel="mAP101 value",
        l1style="-hm",
        l2style="-*g",
        outfile_name="Mask_RCNN_vs_YOLOv5_mAP101_Compare",
    )

    # display barchart AP101 at different IoU thresholds
    mut.output_ap101_barchart_at_iou(mrcnn_lic_metrics, yolo_lic_metrics, 0.55)
    mut.output_ap101_barchart_at_iou(mrcnn_lic_metrics, yolo_lic_metrics, 0.65)
    mut.output_ap101_barchart_at_iou(mrcnn_lic_metrics, yolo_lic_metrics, 0.75)
    mut.output_ap101_barchart_at_iou(mrcnn_lic_metrics, yolo_lic_metrics, 0.85)

    # display barchart: the number of object for each class that the mask
    # r-cnn and yolo (compare with the truth number of object) able to
    # predict with IoU threshold of 0.5 (or 0.8), and confidence threshold
    # of 0.5
    mut.output_tp_barchart_at_iou_conf(
        truth_class_counter, mrcnn_lic_cmatrix, yolo_lic_cmatrix, 0.5, 0.5
    )
    mut.output_tp_barchart_at_iou_conf(
        truth_class_counter, mrcnn_lic_cmatrix, yolo_lic_cmatrix, 0.85, 0.5
    )

    print(f"Evaluation time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
    # main(load_lic_cmatrix=True)
