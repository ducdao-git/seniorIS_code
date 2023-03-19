import os.path as osp

from nuimages import NuImages

from label_mapping import nuim_supported_label_mapping
from truth_class import TruthClass


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

        self.nuim_mrcnn_label_only = nuim_mrcnn_label_only
        if self.nuim_mrcnn_label_only:
            supported_objs_ann = list()

            for obj in self.objs_ann:
                supported_labels = nuim_supported_label_mapping.keys()

                if obj["category_token"] in supported_labels:
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
        if self.nuim_mrcnn_label_only is False:
            print(
                "nuim_mrcnn_label_only set to False. "
                "Thus return get_objs_category from nuimg"
            )
            return self.get_objs_category()

        objs_cat_token = self.get_objs_category_token()
        objs_cat = list()

        for cat_token in objs_cat_token:
            objs_cat.append(nuim_supported_label_mapping[cat_token])

        return objs_cat


def get_truth_objs(nuim_obj, img_index, truth_label_counter):
    # ------------ get the image ------------ #
    sample_img = nuim_obj.sample[img_index]
    sample_img = NuImgSample(
        nuimg=nuim_obj,
        sample_token=sample_img["token"],
        nuim_mrcnn_label_only=True,
    )

    sample_img_path = sample_img.get_sample_img_path()

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

        if truth_obj.t_label not in truth_label_counter.keys():
            truth_label_counter[truth_obj.t_label] = 1
        else:
            truth_label_counter[truth_obj.t_label] += 1

    return sample_img_path, truth_objs
