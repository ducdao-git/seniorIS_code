import json
import time

import pandas as pd

import main_utils as mut
import nuim_util as nuu
from label_mapping import supported_label

NUIMG_INDEX = range(0, 100)
CONFIDENCE_THRESHOLDS = [t / 100 for t in range(0, 101, 10)]
IOU_THRESHOLDS = [t / 100 for t in range(50, 100, 5)]


# CONFIDENCE_THRESHOLDS = [0]
# IOU_THRESHOLDS = [0]


def predict_with_mrcnn(mrcnn_lic_cmatrix, mrcnn_model, img_path, truth_objs):
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
    yolo_preds = mut.get_yolo_predict_objs(yolo_model, img_path)
    # print("-" * 10, "YOLOv5:", len(yolo_preds), "-" * 10)
    print(len(yolo_preds))

    yolo_pt_map, yolo_pt_lmap = mut.get_pred_truth_map(truth_objs, yolo_preds)

    mut.update_lic_cmatrix(
        yolo_lic_cmatrix, yolo_pt_lmap, CONFIDENCE_THRESHOLDS, IOU_THRESHOLDS
    )

    return yolo_lic_cmatrix


def evaluate_model(model_lic_cmatrix, model_name):
    model_lic_metrics = mut.get_lic_metrics(model_lic_cmatrix)
    model_i_mean_ap = mut.get_i_mean_ap(model_lic_metrics)

    model_eval_df = pd.DataFrame(
        {
            "t_iou": model_i_mean_ap["t_ious"],
            "mAP11": model_i_mean_ap["mAP11s"],
            "mAP101": model_i_mean_ap["mAP101s"],
        }
    )
    mut.display_dataframe(
        pandas_dataframe=model_eval_df,
        title=f"{model_name} mAP11 and mAP101 at different IoU thresholds",
        xlabel="IoU threshold",
        ylabel="mAP value",
        outfile_name=f'{"_".join(model_name.split())}_mAP_Evaluation',
    )

    with open(
        f"../output/{'_'.join(model_name.split())}_lic_metrics.json", "w"
    ) as outfile:
        outfile.write(json.dumps(model_lic_metrics))

    return model_eval_df


def main():
    start_time = time.time()

    nuim_obj, mrcnn_model, yolo_model = mut.initialization()

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

    for img_index in NUIMG_INDEX:
        # print("-" * 20, "nuimg_index:", img_index, "-" * 20)
        print("nuimg_index:", img_index, end=", ")

        img_path, truth_objs = nuu.get_truth_objs(nuim_obj, img_index)
        print(len(truth_objs), end=", ")

        mrcnn_lic_cmatrix = predict_with_mrcnn(
            mrcnn_lic_cmatrix, mrcnn_model, img_path, truth_objs
        )

        yolo_lic_cmatrix = predict_with_yolo(
            yolo_lic_cmatrix, yolo_model, img_path, truth_objs
        )

    print(f"Prediction time: {time.time() - start_time}")

    mrcnn_eval_df = evaluate_model(mrcnn_lic_cmatrix, model_name="Mask R-CNN")
    yolo_eval_df = evaluate_model(yolo_lic_cmatrix, model_name="YOLOv5")

    compare_model_df = pd.DataFrame(
        {
            "t_iou": mrcnn_eval_df["t_iou"],
            "Mask R-CNN": mrcnn_eval_df["mAP101"],
            "YOLOv5": yolo_eval_df["mAP101"],
        }
    )
    mut.display_dataframe(
        pandas_dataframe=compare_model_df,
        title="Mask R-CNN's mAP101 vs. YOLOv5's mAP101 at different IoU thresholds",
        xlabel="IoU threshold",
        ylabel="mAP101 value",
        l1style="-hm",
        l2style="-*g",
        outfile_name="Mask_RCNN_vs_YOLOv5_mAP101_Compare",
    )

    print(f"Evaluation time: {time.time() - start_time}")


main()
