import numpy as np


class TruthClass:
    def __init__(self, label, bbox, mask):
        self._label = np.array(label)
        self._bbox = np.array(bbox)
        self._mask = np.array(mask)
        self.is_print_value = True

    @property
    def t_label(self):
        return self._label

    @property
    def t_bbox(self):
        return self._bbox

    @property
    def t_mask(self):
        return self._mask

    def __str__(self):
        label_value, bbox_value, mask_value = ["", "", ""]

        if self.is_print_value:
            label_value += f" {self.t_label}"
            bbox_value += f" {self.t_bbox}"
            mask_value += f" {self.t_mask}"

        return (
            f"t_label: {self.t_label.size}{label_value}\n"
            f"t_bbox: {self.t_bbox.size}{bbox_value}\n"
            f"t_mask: {self.t_mask.size}{mask_value}\n"
        )
