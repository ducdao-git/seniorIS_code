import torch

from label_mapping import supported_label
from predict_class import PredictClass


def get_yolo_predict_objs(
    yolo_model,
    img_path,
    supported_label_only: bool = False,
):
    with torch.no_grad():
        outputs = yolo_model(img_path).pandas().xyxy[0]

    pred_objs = list()
    for _, row in outputs.iterrows():
        if not supported_label_only or row["name"] in supported_label:
            pred_objs.append(
                PredictClass(
                    label=row["name"],
                    score=row["confidence"],
                    bbox=[row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
                    mask=None,
                )
            )

    return pred_objs
