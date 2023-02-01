import matplotlib.pyplot as plt
import numpy as np
import torch
from nuimages import NuImages
from torchvision.io import read_image
from torchvision.transforms import functional as torch_func
from torchvision.utils import draw_bounding_boxes

from mrcnn_utils import get_mask_rcnn, get_mrcnn_outputs
from nuimg_sample import NuImgSample

nuImages_dataroot = "../input/nuImage"
computation_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main(_index):
    # ------------ get the image ------------ #
    nuim = NuImages(
        dataroot=nuImages_dataroot,
        version="v1.0-mini",
        verbose=False,
        lazy=True,
    )

    sample_img = nuim.sample[_index]
    sample_img = NuImgSample(
        nuimg=nuim,
        sample_token=sample_img["token"],
        nuim_mrcnn_label_only=True,
    )

    sample_img_path = sample_img.get_sample_img_path()
    sample_img_int = read_image(sample_img_path)

    # ------------ generate ground truth ------------ #
    t_bboxes = sample_img.get_objs_bbox()  # truth bounding boxes
    t_bboxes = torch.tensor(np.array(t_bboxes), dtype=torch.float)

    t_labels = sample_img.get_objs_mrcnn_category()
    print("--------- truth_label ---------\n", len(t_labels), t_labels)
    print()

    if len(t_bboxes) > 0:
        img = draw_bounding_boxes(
            image=sample_img_int,
            boxes=t_bboxes,
            labels=t_labels,
            colors="red",
            width=1,
        )
    else:
        img = sample_img_int

    # ------------ generate predicted ------------ #
    mrcnn_model, mrcnn_weights = get_mask_rcnn(computation_device)
    transform_sample_img_int = mrcnn_weights.transforms()(sample_img_int)

    p_scores, p_labels, p_bboxes, p_masks = get_mrcnn_outputs(
        mrcnn_model=mrcnn_model,
        mrcnn_weights=mrcnn_weights,
        image=transform_sample_img_int,
        thresh_hold=0.7,
        nuim_mrcnn_label_only=True,
    )

    print("--------- predicted_boxes ---------\n", len(p_bboxes), p_bboxes, "\n")
    print("--------- predicted_labels ---------\n", len(p_labels), p_labels, "\n")
    print("--------- predicted_scores ---------\n", len(p_scores), p_scores, "\n")
    print("--------- predicted_masks ---------\n", len(p_masks), "\n")

    p_bboxes = torch.tensor(np.array(p_bboxes), dtype=torch.float)
    p_labels = [
        label for label in p_labels
        # f"{label}: {score}" for label, score in zip(labels, scores)
    ]

    if len(p_bboxes) > 0:
        img = draw_bounding_boxes(
            image=img,
            boxes=p_bboxes,
            labels=p_labels,
            colors="blue",
            width=1,
        )

    # ------------ show image with bbox ------------ #
    plt.imshow(torch_func.to_pil_image(img))
    plt.show()


# for i in range(20):
main(0)
