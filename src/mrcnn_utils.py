import torch
from torchvision.models import detection as torch_model

from label_mapping import nuim_mrcnn_label_mapping


def get_mask_rcnn(device):
    mrcnn_weights = torch_model.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    # print(f"mrcnn labels: {mrcnn_weights.meta['categories']} \n\n")

    # initialize the model
    mrcnn_model = torch_model.maskrcnn_resnet50_fpn(
        weights=mrcnn_weights,
        progress=True,
        num_classes=91,
    )

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

    # import numpy as np
    # print_masks = torch.tensor(np.array(masks))
    # torch.set_printoptions(profile="full")
    # print(print_masks.size())
    # print(len(boxes))

    # print(print_masks[0][0])

    scores = [score for score in scores if score >= thresh_hold]
    labels = labels[: len(scores)]
    boxes = boxes[: len(scores)]
    masks = masks[: len(scores)]

    if nuim_mrcnn_label_only:
        to_remove_item_index = list()

        for i in range(len(labels)):
            if labels[i] not in nuim_mrcnn_label_mapping.values():
                to_remove_item_index.append(i)

        for index in to_remove_item_index:
            scores.pop(index)
            labels.pop(index)
            boxes.pop(index)
            masks.pop(index)

    return scores, labels, boxes, masks

# def main():
#     import matplotlib.pyplot as plt
#     from torchvision.io import read_image
#     from torchvision.transforms import functional as torch_func
#
#     # set the computation device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     image = read_image(input_img_path)
#
#     mrcnn_model, mrcnn_weights = get_mask_rcnn(device)
#     transform_img = mrcnn_weights.transforms()(image)
#
#     scores, labels, boxes, masks = get_mrcnn_outputs(
#         mrcnn_model, mrcnn_weights, transform_img
#     )
#
#     print("boxes:  ", type(boxes), len(boxes), boxes, "\n")
#     print("labels: ", type(labels), len(labels), labels, "\n")
#     print("scores: ", type(scores), len(scores), scores, "\n")
#     print("masks:  ", type(masks), len(masks), "\n")
#
#     plt.imshow(torch_func.to_pil_image(image))
#     plt.show()
#
#
# main()
