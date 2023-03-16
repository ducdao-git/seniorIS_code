import main_utils as mut
import nuim_util as nuu
import time
from label_mapping import supported_label

NUIMG_INDEX = range(0, 2)
# CONFIDENCE_THRESHOLDS = [t / 100 for t in range(0, 11, 10)]
# IOU_THRESHOLDS = [t / 100 for t in range(50, 66, 5)]
CONFIDENCE_THRESHOLDS = [0]
IOU_THRESHOLDS = [0]


def main():
    nuim_obj, mrcnn_model, yolo_model = mut.initialization()

    # lic_cmatrix -- a dict store the confusion matrix based on label, iou
    # threshold, and confidence threshold
    lic_cmatrix = mut.init_lic_cmatrix(
        supported_labels=supported_label,
        t_ious=IOU_THRESHOLDS,
        t_confidences=CONFIDENCE_THRESHOLDS,
    )

    for img_index in NUIMG_INDEX:
        print("-" * 20, "nuimg_index:", img_index, "-" * 20)
        img_path, truth_objs = nuu.get_truth_objs(nuim_obj, img_index)

        mrcnn_preds = mut.get_mrcnn_predict_objs(mrcnn_model, img_path)
        print("-" * 10, "MRCNN:", len(mrcnn_preds), "-" * 10)

        mr_pt_map, mr_pt_lmap = mut.get_pred_truth_map(truth_objs, mrcnn_preds)
        mut.update_lic_cmatrix(
            lic_cmatrix, mr_pt_lmap, CONFIDENCE_THRESHOLDS, IOU_THRESHOLDS
        )

        # yolo_preds = mut.get_yolo_predict_objs(yolo_model, img_path)
        # print("-" * 10, "YOLOv5:", len(yolo_preds), "-" * 10)
        # # for obj in yolo_preds:
        # #     print(obj)


main()
