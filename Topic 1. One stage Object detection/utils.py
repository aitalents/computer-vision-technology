import torch


def compute_pairwise_iou(bboxes1, bboxes2):
    """
    Returns ious tensor with IoU values computed pairwise
    for all corresponding bboxes from input arrays.
    Input bboxes in format (x_center, y_center, w, h, confidence).
    """
    x_min_1, y_min_1 = bboxes1[:, 0] - bboxes1[:, 2] * 0.5, bboxes1[:, 1] - bboxes1[:, 3] * 0.5
    x_max_1, y_max_1 = bboxes1[:, 0] + bboxes1[:, 2] * 0.5, bboxes1[:, 1] + bboxes1[:, 3] * 0.5

    x_min_2, y_min_2 = bboxes2[:, 0] - bboxes2[:, 2] * 0.5, bboxes2[:, 1] - bboxes2[:, 3] * 0.5
    x_max_2, y_max_2 = bboxes2[:, 0] + bboxes2[:, 2] * 0.5, bboxes2[:, 1] + bboxes2[:, 3] * 0.5

    area1 = bboxes1[:, 2] * bboxes1[:, 3]
    area2 = bboxes2[:, 2] * bboxes2[:, 3]

    zero = torch.zeros(x_min_1.size())

    inter_width = torch.max(zero, torch.min(x_max_1, x_max_2) - torch.max(x_min_1,x_min_2))
    inter_height = torch.max(zero, torch.min(y_max_1, y_max_2) - torch.max(y_min_1,y_min_2))
    inter_area = inter_width * inter_height
    union_area = (area1 + area2) - inter_area

    return inter_area / union_area
