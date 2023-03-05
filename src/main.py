import matplotlib.pyplot as plt
import numpy as np
import torch
from nuimages import NuImages
from torchvision.io import read_image
from torchvision.transforms import functional as torch_func
from torchvision.utils import draw_bounding_boxes


from metrics import preds_choose_truths_map, get_box_iou
from mrcnn_utils import get_mask_rcnn, get_mrcnn_outputs
from nuimg_sample import NuImgSample

from truth_class import TruthClass
from predict_class import PredictClass

from pprint import pprint

# torch.set_printoptions(profile="full")
# torch.set_printoptions(linewidth=10000)

nuImages_dataroot = "../input/nuImage"
nuim_obj = NuImages(
    dataroot=nuImages_dataroot,
    version="v1.0-mini",
    verbose=False,
    lazy=True,
)
computation_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_truth_objs(img_index):
    # ------------ get the image ------------ #
    sample_img = nuim_obj.sample[img_index]
    sample_img = NuImgSample(
        nuimg=nuim_obj,
        sample_token=sample_img["token"],
        nuim_mrcnn_label_only=True,
    )

    sample_img_path = sample_img.get_sample_img_path()
    sample_img_int = read_image(sample_img_path)

    # ----------------- generate ground truth ----------------- #
    t_bboxes = sample_img.get_objs_bbox()
    t_labels = sample_img.get_objs_mrcnn_category()
    t_masks = sample_img.get_objs_mask()

    truth_objs = list()
    for i in range(len(t_labels)):
        truth_obj = TruthClass(
            label=t_labels[i],
            bbox=t_bboxes[i],
            mask=t_masks[i],
        )
        truth_objs.append(truth_obj)

    return sample_img_int, truth_objs


def get_mrcnn_predict_objs(sample_img_int, thresh_hold=0.7):
    # ----------------- generate predicted ----------------- #
    mrcnn_model, mrcnn_weights = get_mask_rcnn(computation_device)
    transform_sample_img_int = mrcnn_weights.transforms()(sample_img_int)

    p_scores, p_labels, p_bboxes, p_masks = get_mrcnn_outputs(
        mrcnn_model=mrcnn_model,
        mrcnn_weights=mrcnn_weights,
        image=transform_sample_img_int,
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

    return transform_sample_img_int, pred_objs


def main(_index):
    sample_img_int, truth_objs = get_truth_objs(_index)
    sample_img_int, pred_objs = get_mrcnn_predict_objs(sample_img_int, 0.7)

    # ----------------- evaluation based on bboxes ----------------- #
    pred_truth_map = preds_choose_truths_map(
        truth_objs=truth_objs,
        pred_objs=pred_objs,
        threshold=0.9,
    )

    pprint(pred_truth_map)
    # print(pred_truth_map[0]["iou_score"])
    # print(pred_truth_map[0]["pred_obj"])
    # print(pred_truth_map[0]["truth_obj"])

    # # ----------------- show image with bbox ----------------- #
    # t_bboxes = torch.tensor(np.array(t_bboxes), dtype=torch.float)
    # if len(t_bboxes) > 0:
    #     img = draw_bounding_boxes(
    #         image=sample_img_int,
    #         boxes=t_bboxes,
    #         labels=t_labels,
    #         colors="red",
    #         width=1,
    #     )
    # else:
    #     img = sample_img_int
    #
    # p_bboxes = torch.tensor(np.array(p_bboxes), dtype=torch.float)
    # if len(p_bboxes) > 0:
    #     img = draw_bounding_boxes(
    #         image=img,
    #         boxes=p_bboxes,
    #         labels=p_labels,
    #         colors="blue",
    #         width=1,
    #     )
    #
    # plt.imshow(torch_func.to_pil_image(img))
    # plt.show()


# for i in range(20):
main(0)
