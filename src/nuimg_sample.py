import os.path as osp

from nuimages import NuImages

from label_mapping import nuim_mrcnn_label_mapping


class NuImgSample:
    def __init__(
        self,
        nuimg: NuImages,
        sample_token: str,
        nuim_mrcnn_label_only: bool = False,
    ):
        self.nuimg = nuimg
        self.sample_img = self.nuimg.get(
            table_name="sample", token=sample_token
        )

        # get annotation data for all object that present in sample_img,
        # where the sample image is identified by key_camera_token
        self.objs_ann = [
            nuimg.get(table_name="object_ann", token=obj_token)
            for obj_token in self.nuimg.list_anns(
                sample_token=self.sample_img["token"],
                verbose=False,
            )[0]
        ]

        if nuim_mrcnn_label_only:
            supported_objs_ann = list()

            for obj in self.objs_ann:
                if obj["category_token"] in nuim_mrcnn_label_mapping.keys():
                    supported_objs_ann.append(obj)

            self.objs_ann = supported_objs_ann

    def get_sample_img(self):
        return self.sample_img.copy()

    def get_sample_img_path(self):
        sample_data = self.nuimg.get("sample_data", self.get_keycam_token())
        img_path = osp.join(self.nuimg.dataroot, sample_data["filename"])

        return img_path

    def get_keycam_token(self):
        return self.sample_img["key_camera_token"]

    def get_objs_ann(self):
        return [obj for obj in self.objs_ann]

    def get_objs_bbox(self):
        return [obj["bbox"] for obj in self.objs_ann]

    def get_objs_mask(self):
        return [obj["mask"] for obj in self.objs_ann]

    def get_objs_category_token(self):
        return [obj["category_token"] for obj in self.objs_ann]

    def get_objs_category(self):
        objs_cat_token = self.get_objs_category_token()
        objs_cat = list()

        for cat_token in objs_cat_token:
            objs_cat.append(
                self.nuimg.get(table_name="category", token=cat_token)["name"]
            )

        return objs_cat

    def get_objs_mrcnn_category(self):
        objs_cat_token = self.get_objs_category_token()
        objs_cat = list()

        for cat_token in objs_cat_token:
            objs_cat.append(nuim_mrcnn_label_mapping[cat_token])

        return objs_cat


# --------------------------------------------------------------------------- #
# def main():
#     import matplotlib.pyplot as plt
#     from nuimages import NuImages
#     from torchvision.io import read_image
#     from torchvision.transforms import functional as torch_func
#
#     nuim = NuImages(
#         dataroot="../input/nuImage",
#         version="v1.0-mini",
#         verbose=False,
#         lazy=True,
#     )
#
#     sample_0 = nuim.sample[0]
#     sample_0 = NuImgSample(nuimg=nuim, sample_token=sample_0['token'])
#
#     sample_0_path = sample_0.get_sample_img_path()
#     sample_0_int = read_image(sample_0_path)
#
#     plt.imshow(torch_func.to_pil_image(sample_0_int))
#     plt.show()
#
#
# main()
#
# import torch

#
# import torchvision.transforms.functional as F
#
# from torchvision.utils import draw_bounding_boxes
#
# from torchvision.io import read_image
#
# scene_int = read_image(im_path)
#
# boxes = torch.tensor(obj_bbox_in_img, dtype=torch.float)
# colors = ["blue"] * len(obj_bbox_in_img)
# result = draw_bounding_boxes(scene_int, boxes, colors=colors, width=5)
#
# plt.imshow(np.asarray(F.to_pil_image(result)))
# plt.show()
