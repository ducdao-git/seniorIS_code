import torch
from torchvision.ops import box_iou

# jaccard = JaccardIndex(task="multiclass", num_classes=3)
#
# t_bboxes = torch.tensor([
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 2],
# ])
#
# p_bboxes = torch.tensor([
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 2],
# ])
#
# # p_bboxes = torch.tensor(
# #     [
# #         [
# #             [
# #                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
# #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
# #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
# #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
# #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
# #                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
# #             ],
# #         ]
# #     ]
# # )
#
# print("score =", jaccard(preds=p_bboxes, target=t_bboxes))

t_boxes = torch.tensor([[1, 1, 2, 2], [0, 0, 1, 1]])
p_boxes = torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3]])

print(box_iou(
    boxes1=t_boxes,
    boxes2=p_boxes,
))
