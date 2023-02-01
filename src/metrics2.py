import torch
from torchmetrics import JaccardIndex

jaccard = JaccardIndex(task="multiclass", num_classes=3)

t_bboxes = torch.tensor([
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
])

p_bboxes = torch.tensor([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
    [0, 0, 1, 1, 2, 2],
])

# print(t_bboxes)
# print()
# print(p_bboxes)

print(jaccard(preds=p_bboxes, target=t_bboxes))
