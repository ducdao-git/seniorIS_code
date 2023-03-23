import numpy as np


class TruthObject:
    def __init__(self, label, bbox, mask):
        self._label = str(label)
        self._bbox = np.array(bbox)
        self._mask = np.array(mask)
        self.is_print_value = False

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
        bbox_value, mask_value = ["", ""]

        if self.is_print_value:
            bbox_value += f" {self.t_bbox}"
            mask_value += f" {self.t_mask}"

        return (
            f"t_label: {self.t_label}\n"
            f"t_bbox: {self.t_bbox.size}{bbox_value}\n"
            f"t_mask: {self.t_mask.size}{mask_value}\n"
        )
