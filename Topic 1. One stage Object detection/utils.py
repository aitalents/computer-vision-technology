from PIL import ImageDraw
import torch
from torchvision.transforms import ToPILImage


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


def nms(img_bboxes_list, num_classes, iou_thr=0.45):
    results = []

    for cls in range(num_classes):
        cls_boxes = [box for box in img_bboxes_list if box[-1] == cls]

        if not len(cls_boxes):
            continue

        sorted_boxes = sorted(cls_boxes, key=lambda x: x[5], reverse=True)
        cls_res = [sorted_boxes[0]]

        for i in range(len(cls_boxes) - 1):
            bbox_coords = torch.tensor(sorted_boxes[i + 1]).reshape((1, 6))
            used_boxes = torch.tensor(cls_res).reshape((-1, 6))

            ious = compute_pairwise_iou(used_boxes, bbox_coords)
            max_iou = torch.max(ious)

            if max_iou <= iou_thr:
                cls_res.append(sorted_boxes[i + 1])

        results.extend(cls_res)

    return results


def render_pred_bboxes(img, pred_bboxes):
    img_h, img_w, _ = img.size()

    img = ToPILImage()(img)
    img1 = ImageDraw.Draw(img)

    for pred_bbox in pred_bboxes:
        x_center, y_center, w, h, conf, class_id = pred_bbox
        x0, y0 = x_center - w * 0.5, y_center - h * 0.5
        x1, y1 = x_center + w * 0.5, y_center + h * 0.5
        rect_coords = [
            (int(x0 * img_w), int(y0 * img_h)),
            (int(x1 * img_w), int(y1 * img_h)),
        ]
        img1.rectangle(rect_coords, outline ="green")
        img1.text(rect_coords[0], f'cl:{class_id},conf:{conf:.2f}', (0,255,0))

    img.show()

    return img
