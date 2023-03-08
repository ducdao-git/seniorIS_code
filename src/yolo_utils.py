import torch

from label_mapping import label_int_map
from predict_class import PredictClass

YOLO_MODEL = torch.hub.load(
    repo_or_dir="ultralytics/yolov5",
    model="yolov5x",
    pretrained=True,
    verbose=False,
)


def get_yolov5_predict_objs(computation_device, sample_img_path):
    model = YOLO_MODEL

    # model = torch.hub.load(
    #     'ultralytics/yolov5', 'custom', 'yolov5m-seg.pt', force_reload=True
    # )
    model.to(computation_device).eval()

    with torch.no_grad():
        outputs = model(sample_img_path)

    dataframe = outputs.pandas().xyxy[0]
    # pprint(dataframe)
    # print()

    pred_objs = list()
    for _, row in dataframe.iterrows():
        if row["name"] in label_int_map:
            pred_objs.append(
                PredictClass(
                    label=row["name"],
                    score=row["confidence"],
                    bbox=[row["xmin"], row["ymin"], row["xmax"], row["ymax"]],
                    mask=None,
                )
            )

    return pred_objs
