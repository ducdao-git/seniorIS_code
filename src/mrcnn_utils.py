import torch

from label_mapping import supported_label
from predict_object import PredictObject


def get_mrcnn_outputs(
    mrcnn_model,
    mrcnn_weights,
    img_tensor,
    supported_label_only: bool = False,
):
    with torch.no_grad():
        outputs = mrcnn_model([img_tensor])

    scores = list(outputs[0]["scores"].detach().cpu().numpy())
    labels = [
        mrcnn_weights.meta["categories"][label_id]
        for label_id in outputs[0]["labels"]
    ]

    boxes = list(outputs[0]["boxes"].detach().cpu().numpy())
    masks = list(outputs[0]["masks"].detach().cpu().numpy())
    # masks = list(outputs[0]["masks"].squeeze().detach().cpu().numpy())

    if not supported_label_only:
        return labels, scores, boxes, masks

    remove_index = set()
    for i in range(len(labels)):
        if labels[i] not in supported_label:
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

    return return_labels, return_scores, return_boxes, return_masks


def mrcnn_output_to_objs(labels, scores, boxes, masks):
    pred_objs = list()
    for i in range(len(labels)):
        pred_obj = PredictObject(
            label=labels[i],
            score=scores[i],
            bbox=boxes[i],
            mask=masks[i],
        )
        pred_objs.append(pred_obj)

    return pred_objs
