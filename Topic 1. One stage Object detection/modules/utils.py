from collections import Counter

import torch
import torch.nn as nn


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=80, eps=1e-6):
    average_precisions = []
    for class_index in range(num_classes):
        detections = [det for det in pred_boxes if det[1] == class_index]
        ground_truths = [
            true_box for true_box in true_boxes if true_box[1] == class_index
        ]
        num_ground_truths = len(ground_truths)
        if num_ground_truths == 0:
            continue
        detected = Counter([gt[0] for gt in ground_truths])
        for key in detected:
            detected[key] = torch.zeros(detected[key])
        detections.sort(key=lambda x: x[2], reverse=True)
        true_positive = torch.zeros((len(detections)))
        false_positive = torch.zeros((len(detections)))
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            best_iou = 0
            for i, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), torch.tensor(gt[3:])
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_i = i
            if best_iou > iou_threshold:
                if detected[detection[0]][best_gt_i] == 0:
                    true_positive[detection_idx] = 1
                    detected[detection[0]][best_gt_i] = 1
                else:
                    false_positive[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1
        tp_cumsum = torch.cumsum(true_positive, dim=0)
        fp_cumsum = torch.cumsum(false_positive, dim=0)
        recalls = tp_cumsum / (num_ground_truths + eps)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)


def intersection_over_union(boxes_preds, boxes_labels):
    p_box_corner1 = boxes_preds[..., :2] - boxes_preds[..., 2:4] / 2
    p_box_corner2 = boxes_preds[..., :2] + boxes_preds[..., 2:4] / 2
    true_box_corner1 = boxes_labels[..., :2] - boxes_labels[..., 2:4] / 2
    true_box_corner2 = boxes_labels[..., :2] + boxes_labels[..., 2:4] / 2
    intersect_mins = torch.max(p_box_corner1, true_box_corner1)
    intersect_maxes = torch.min(p_box_corner2, true_box_corner2)
    intersect_wh = torch.clamp(intersect_maxes - intersect_mins, min=0)
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]
    p_box_area = abs(
        (p_box_corner2[..., 0] - p_box_corner1[..., 0])
        * (p_box_corner2[..., 1] - p_box_corner1[..., 1])
    )
    true_box_area = abs(
        (true_box_corner2[..., 0] - true_box_corner1[..., 0])
        * (true_box_corner2[..., 1] - true_box_corner1[..., 1])
    )
    return intersection / (p_box_area + true_box_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, prob_threshold):
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes.sort(key=lambda x: x[1], reverse=True)
    nms_boxes = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            )
            < iou_threshold
        ]
        nms_boxes.append(chosen_box)
    return nms_boxes


class YoloDetectionLoss(nn.Module):
    def __init__(self, S=7, B=2, C=80):
        super(YoloDetectionLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        ious = self.compute_ious(predictions, target)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)
        box_predictions = self.get_box_predictions(exists_box, bestbox, predictions)
        box_targets = exists_box * target[..., self.C + 1 : self.C + 5]
        box_predictions, box_targets = self.apply_sqrt_width_height(
            box_predictions, box_targets
        )
        box_loss = self.compute_box_loss(box_predictions, box_targets)
        p_box = (bestbox * predictions[..., self.C + 5 : self.C + 6]
                 + (1 - bestbox) * predictions[..., self.C : self.C + 1]
        )
        object_loss = self.compute_object_loss(exists_box, p_box, target)
        no_object_loss = self.compute_no_object_loss(exists_box, predictions, target)
        class_loss = self.compute_class_loss(exists_box, predictions, target)
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        return loss

    def compute_ious(self, predictions, target):
        iou_b1 = intersection_over_union(
            predictions[..., self.C + 1 : self.C + 5],
            target[..., self.C + 1 : self.C + 5],
        )
        iou_b2 = intersection_over_union(
            predictions[..., self.C + 6 : self.C + 10],
            target[..., self.C + 1 : self.C + 5],
        )
        return torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

    def get_box_predictions(self, exists_box, bestbox, predictions):
        return exists_box * (
            bestbox * predictions[..., self.C + 6 : self.C + 10]
            + (1 - bestbox) * predictions[..., self.C + 1 : self.C + 5]
        )

    def apply_sqrt_width_height(self, box_predictions, box_targets):
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        return box_predictions, box_targets

    def compute_box_loss(self, box_predictions, box_targets):
        return self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

    def compute_object_loss(self, exists_box, p_box, target):
        return self.mse(
            torch.flatten(exists_box * p_box),
            torch.flatten(exists_box * target[..., self.C : self.C + 1]),
        )

    def compute_no_object_loss(self, exists_box, predictions, target):
        no_object_loss = self.mse(
            torch.flatten(
                (1 - exists_box) * predictions[..., self.C : self.C + 1], start_dim=1
            ),
            torch.flatten(
                (1 - exists_box) * target[..., self.C : self.C + 1], start_dim=1
            ),
        )
        no_object_loss += self.mse(
            torch.flatten(
                (1 - exists_box) * predictions[..., self.C + 5 : self.C + 6],
                start_dim=1,
            ),
            torch.flatten(
                (1 - exists_box) * target[..., self.C : self.C + 1], start_dim=1
            ),
        )
        return no_object_loss

    def compute_class_loss(self, exists_box, predictions, target):
        return self.mse(
            torch.flatten(exists_box * predictions[..., : self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., : self.C], end_dim=-2),
        )


def convert_cells(predictions, S=7, C=80):
    predictions = predictions.to("cpu").reshape(-1, S, S, C + 10)
    bboxes1 = predictions[..., C + 1 : C + 5]
    bboxes2 = predictions[..., C + 6 : C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(predictions.shape[0], S, 1).unsqueeze(-1)
    converted_bboxes = torch.cat(
        (1 / S * (best_boxes[..., :2] + cell_indices), 1 / S * best_boxes[..., 2:4]),
        dim=-1,
    )
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(
        -1
    )
    return torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)


def cells_to_boxes(out, S=7):
    converted_pred = convert_cells(out).reshape(out.shape[0], S * S, -1)
    return [
        [x.item() for x in converted_pred[ex_idx, bbox_idx, :]]
        for ex_idx in range(out.shape[0])
        for bbox_idx in range(S * S)
    ]


def extract_bboxes(loader, model, iou_threshold, prob_threshold, device="cuda"):
    all_p_boxes = []
    all_gt_boxes = []
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(images)
        batch_size = images.shape[0]
        true_bboxes = cells_to_boxes(labels)
        p_bboxes = cells_to_boxes(predictions)
        for j in range(batch_size):
            nms_boxes = non_max_suppression(
                p_bboxes[j],
                iou_threshold=iou_threshold,
                prob_threshold=prob_threshold,
            )
            for nms_box in nms_boxes:
                all_p_boxes.append([i] + nms_box)
            for true_box in true_bboxes[j]:
                if true_box[1] > prob_threshold:
                    all_gt_boxes.append([i] + true_box)
    return all_p_boxes, all_gt_boxes
