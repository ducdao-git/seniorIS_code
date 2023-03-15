import torch
from torchvision.io import read_image
from torchvision.models import detection as torch_model

from label_mapping import nuim_supported_label_mapping
from predict_class import PredictClass

MRCNN_WEIGHTS = torch_model.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
MRCNN_MODEL = torch_model.maskrcnn_resnet50_fpn(
    weights=MRCNN_WEIGHTS,
    progress=True,
    num_classes=91,
)


def get_mask_rcnn(device):
    mrcnn_weights = MRCNN_WEIGHTS
    mrcnn_model = MRCNN_MODEL

    # load the model on to the computation device and set to eval mode
    mrcnn_model.to(device).eval()

    return mrcnn_model, mrcnn_weights


def get_mrcnn_outputs(
    mrcnn_model,
    mrcnn_weights,
    image,
    thresh_hold=0.95,
    nuim_mrcnn_label_only: bool = False,
):
    with torch.no_grad():
        # pass of the image to the model without gradient calculation step
        outputs = mrcnn_model([image])

    scores = list(outputs[0]["scores"].detach().cpu().numpy())
    labels = [
        mrcnn_weights.meta["categories"][label_id]
        for label_id in outputs[0]["labels"]
    ]

    boxes = list(outputs[0]["boxes"].detach().cpu().numpy())
    masks = list(outputs[0]["masks"].detach().cpu().numpy())
    # masks = list(outputs[0]["masks"].squeeze().detach().cpu().numpy())

    scores = [score for score in scores if score >= thresh_hold]
    labels = labels[: len(scores)]
    boxes = boxes[: len(scores)]
    masks = masks[: len(scores)]

    if nuim_mrcnn_label_only:
        remove_index = set()

        for i in range(len(labels)):
            if labels[i] not in nuim_supported_label_mapping.values():
                remove_index.add(i)

        return_scores = list()
        return_labels = list()
        return_boxes = list()
        return_masks = list()
        for i in range(len(labels)):
            if i in remove_index:
                continue

            return_scores.append(scores[i])
            return_labels.append(labels[i])
            return_boxes.append(boxes[i])
            return_masks.append(masks[i])

        return return_scores, return_labels, return_boxes, return_masks

    return scores, labels, boxes, masks


def get_mrcnn_predict_objs(
    computation_device, sample_img_path, thresh_hold=0.5
):
    # ----------------- generate predicted ----------------- #
    mrcnn_model, mrcnn_weights = get_mask_rcnn(computation_device)

    sample_img_int = read_image(sample_img_path)
    transform_sample_img_int = mrcnn_weights.transforms()(sample_img_int)

    p_scores, p_labels, p_bboxes, p_masks = get_mrcnn_outputs(
        mrcnn_model=mrcnn_model,
        mrcnn_weights=mrcnn_weights,
        image=transform_sample_img_int.to(computation_device),
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

    return pred_objs
