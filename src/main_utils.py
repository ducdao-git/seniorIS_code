import torch
from nuimages import NuImages
from torchvision.io import read_image
from torchvision.models import detection as torch_model

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

    mrcnn_model = torch_model.maskrcnn_resnet50_fpn(
        weights=MRCNN_WEIGHTS,
        progress=True,
        num_classes=91,
    ).to(PROCESSOR).eval()

    yolo_model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5",
        model="yolov5x",
        pretrained=True,
        verbose=False,
    ).to(PROCESSOR).eval()

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
