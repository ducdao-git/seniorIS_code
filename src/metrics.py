def get_box_iou(t_box_cord, p_box_cord):
    p_box = {
        "x1": p_box_cord[0],
        "y1": p_box_cord[1],
        "x2": p_box_cord[2],
        "y2": p_box_cord[3],
    }

    t_box = {
        "x1": t_box_cord[0],
        "y1": t_box_cord[1],
        "x2": t_box_cord[2],
        "y2": t_box_cord[3],
    }

    if p_box["x1"] > p_box["x2"] or p_box["y1"] > p_box["y2"]:
        raise ValueError("p_box_cord must in (xmin, ymin, xmax, ymax) format")

    if t_box["x1"] > t_box["x2"] or t_box["y1"] > t_box["y2"]:
        raise ValueError("t_box_cord must in (xmin, ymin, xmax, ymax) format")

    # coordinate of the intersection rectangle
    inter_x1 = max(p_box["x1"], t_box["x1"])
    inter_y1 = max(p_box["y1"], t_box["y1"])
    inter_x2 = min(p_box["x2"], t_box["x2"])
    inter_y2 = min(p_box["y2"], t_box["y2"])

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0

    # area of the intersection rectangle
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # area of predicted box and truth box
    p_box_area = (p_box["x2"] - p_box["x1"]) * (p_box["y2"] - p_box["y1"])
    t_box_area = (t_box["x2"] - t_box["x1"]) * (t_box["y2"] - t_box["y1"])

    return intersection_area / (p_box_area + t_box_area - intersection_area)


assert get_box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1
assert get_box_iou([1, 0, 2, 2], [0, 0, 2, 2]) == 0.5
assert get_box_iou([1, 1, 2, 2], [0, 0, 2, 2]) == 0.25

assert get_box_iou([0, 1, 2, 2], [0, 0, 1, 1]) == 0
assert get_box_iou([2, 2, 3, 3], [0, 0, 1, 1]) == 0

# the get_box_iou is not perfect, let say we have 2 truth boxes, one with
# (0, 0, 4, 4) and one with (0, 2, 4, 4). If the predict box is
# (0, 1, 4, 4), then iou will higher for pair (0, 0, 4, 4) compare to
# (0, 2, 4, 4). The math make sense, but in reality, is this the result that
# we want?

print(get_box_iou([0, 0, 4, 4], [0, 1, 4, 4]))
print(get_box_iou([0, 2, 4, 4], [0, 1, 4, 4]))