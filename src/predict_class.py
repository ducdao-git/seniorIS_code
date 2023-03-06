import numpy as np


class PredictClass:
    def __init__(self, label, score, bbox, mask):
        self._label = str(label)
        self._score = np.array(score)
        self._bbox = np.array(bbox)
        self._mask = np.array(mask)
        self.is_print_value = True

    @property
    def p_label(self):
        return self._label

    @property
    def p_score(self):
        return self._score

    @property
    def p_bbox(self):
        return self._bbox

    @property
    def p_mask(self):
        return self._mask

    def __str__(self):
        label_value, score_value, bbox_value, mask_value = ["", "", "", ""]

        if self.is_print_value:
            label_value += f" {self.p_label}"
            score_value += f" {self.p_score}"
            bbox_value += f" {self.p_bbox}"
            mask_value += f" {self.p_mask}"

        return (
            f"p_label: {label_value}\n"
            f"p_score: {self.p_score.size}{score_value}\n"
            f"p_bbox: {self.p_bbox.size}{bbox_value}\n"
            f"p_mask: {self.p_mask.size}{mask_value}\n"
        )
