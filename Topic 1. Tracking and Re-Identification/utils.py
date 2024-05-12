import torch
from collections import Counter


class AnchorBoxes:
    def __init__(self, anchor_sizes, grid_size, image_size):
        self.anchor_sizes = torch.tensor(anchor_sizes, dtype=torch.float32)
        self.grid_size = grid_size
        self.stride = image_size / grid_size
        self.anchors = self._create_anchors()

    def _create_anchors(self):
        # Создание якорей для каждой ячейки сетки.
        cell_indices = torch.arange(self.grid_size)
        x, y = torch.meshgrid(cell_indices, cell_indices, indexing='ij')
        # Вычисление центров ячеек
        cell_centers = torch.stack((x, y), dim=2) * self.stride + self.stride / 2
        cell_centers = cell_centers.view(-1, 1, 2).expand(-1, len(self.anchor_sizes), 2)
        anchors = torch.cat((cell_centers, self.anchor_sizes.expand(self.grid_size**2, -1, -1)), dim=2)
        # grid_size, grid_size, num_anchors, 4
        return anchors.view(self.grid_size, self.grid_size, -1, 4)

    def assign_anchors(self, bboxes, calculate_iou):
        # Назначение каждой рамке наилучшего якоря на основе IoU.
        num_bboxes = bboxes.shape[0]
        anchors_flat = self.anchors.view(-1, 4)
        # Вычисляем IoU между каждой рамкой и якорем
        ious = calculate_iou(bboxes, anchors_flat)
        # Индекс якоря с максимальным IoU для каждой рамки
        best_anchors = ious.argmax(dim=1)  
        # Позиции на сетке и индексы якорей
        grid_positions = best_anchors // (len(self.anchor_sizes) * self.grid_size)
        anchor_indices = best_anchors % len(self.anchor_sizes)
        return torch.stack([grid_positions // self.grid_size, grid_positions % self.grid_size, anchor_indices], dim=1)



class DetectionUtils:
    # Фильтрация и сортировка ограничивающих рамок по порогу уверенности и IoU
    def filter_and_sort_boxes(self, boxes, score_threshold, iou_threshold, format="corners"):
        assert isinstance(boxes, list), "Boxes должны быть списком"

        # Отфильтровываем рамки, уверенность которых ниже порога
        boxes = [box for box in boxes if box[1] > score_threshold]
        # Сортируем рамки по уверенности в убывающем порядке
        boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
        filtered_boxes = []

        # Пока есть рамки для обработки
        while boxes:
            current_box = boxes.pop(0)
            # Оставляем только те рамки, которые имеют IoU меньше заданного порога
            boxes = [
                box for box in boxes
                if box[0] != current_box[0]
                or self.calculate_iou(
                    torch.tensor(current_box[2:]),
                    torch.tensor(box[2:]),
                    format=format,
                ) < iou_threshold
            ]
            # Добавляем выбранную рамку в результат
            filtered_boxes.append(current_box)

        return filtered_boxes

    # Расчет IoU (пересечение по объединению) для двух рамок
    def calculate_iou(self, box1, box2, format='midpoint'):
        # Переводим координаты рамок в зависимости от формата
        if format == 'midpoint':
            box1 = torch.cat([box1[..., :2] - box1[..., 2:] / 2, box1[..., :2] + box1[..., 2:] / 2], dim=-1)
            box2 = torch.cat([box2[..., :2] - box2[..., 2:] / 2, box2[..., :2] + box2[..., 2:] / 2], dim=-1)
        
        # Вычисляем пересечение рамок
        intersect_min = torch.max(box1[:, :2], box2[:, :2])
        intersect_max = torch.min(box1[:, 2:], box2[:, 2:])
        intersect_area = (intersect_max - intersect_min).clamp(min=0).prod(dim=1)

        # Вычисляем площадь каждой рамки
        box1_area = (box1[:, 2:] - box1[:, :2]).prod(dim=1)
        box2_area = (box2[:, 2:] - box2[:, :2]).prod(dim=1)

        # Возвращаем IoU
        return intersect_area / (box1_area + box2_area - intersect_area + 1e-6)

    # Расчет средней точности для всех классов
    def calculate_map(self, pred_boxes, true_boxes, iou_threshold=0.5, format="midpoint", num_classes=91):
        average_precisions = []
        epsilon = 1e-6

        # Обходим все классы
        for class_index in range(num_classes):
            # Отбираем детекции и истинные значения для текущего класса
            class_detections = [det for det in pred_boxes if det[1] == class_index]
            class_ground_truths = [gt for gt in true_boxes if gt[1] == class_index]
            gt_count = Counter(gt[0] for gt in class_ground_truths)
            gt_count = {key: torch.zeros(val) for key, val in gt_count.items()}
            class_detections.sort(key=lambda x: x[2], reverse=True)
            
            TP = torch.zeros(len(class_detections))
            FP = torch.zeros(len(class_detections))
            total_gts = len(class_ground_truths)

            if total_gts == 0:
                continue

            # Вычисляем TP и FP для каждой детекции
            for idx, detection in enumerate(class_detections):
                ground_truths_img = [gt for gt in class_ground_truths if gt[0] == detection[0]]
                best_iou, best_gt_idx = 0, -1

                for gt_idx, gt in enumerate(ground_truths_img):
                    iou = self.calculate_iou(torch.tensor(detection[3:]), torch.tensor(gt[3:]), format=format)
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, gt_idx

                if best_iou > iou_threshold and gt_count[detection[0]][best_gt_idx] == 0:
                    TP[idx] = 1
                    gt_count[detection[0]][best_gt_idx] = 1
                else:
                    FP[idx] = 1

            TP_cum = torch.cumsum(TP, dim=0)
            FP_cum = torch.cumsum(FP, dim=0)
            precisions = TP_cum / (TP_cum + FP_cum + epsilon)
            recalls = TP_cum / (total_gts + epsilon)
            average_precisions.append(torch.trapz(torch.cat([torch.tensor([1]), precisions]), torch.cat([torch.tensor([0]), recalls])))

        # Возвращаем среднюю точность по всем классам
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0

    # Получение отфильтрованных рамок из данных загрузчика
    def get_filtered_boxes(self, loader, model, iou_threshold, score_threshold, format="midpoint", device="cuda"):
        model.eval()
        all_predictions, all_ground_truths = [], []
        train_index = 0

        for _, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                predictions = model(inputs)

            for idx in range(inputs.size(0)):
                filtered_preds = self.filter_and_sort_boxes(predictions[idx], score_threshold, iou_threshold, format=format)
                filtered_gts = [box for box in self.convert_cellboxes(labels[idx]) if box[1] > score_threshold]
                
                all_predictions.extend([[train_index] + box for box in filtered_preds])
                all_ground_truths.extend([[train_index] + box for box in filtered_gts])

                train_index += 1

        model.train()
        return all_predictions, all_ground_truths

    # Конвертация предсказаний ячеек в ограничивающие рамки
    def convert_cellboxes(self, predictions, grid_size=7, num_classes=91):
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, grid_size, grid_size, num_classes + 10)
        bboxes = torch.cat((predictions[..., num_classes+1:num_classes+5], predictions[..., num_classes+6:num_classes+10]), dim=-1)
        scores = torch.cat((predictions[..., num_classes:num_classes+1], predictions[..., num_classes+5:num_classes+6]), dim=-1).max(dim=-1, keepdim=True)[0]
        best_boxes = (bboxes[..., :4] * (scores < 0.5).float() + bboxes[..., 4:] * (scores >= 0.5).float())
        cell_indices = torch.arange(grid_size).repeat(batch_size, grid_size, 1).unsqueeze(-1)
        normalized_boxes = torch.cat((cell_indices + best_boxes[..., :2], best_boxes[..., 2:]), dim=-1) / grid_size
        predicted_classes = predictions[..., :num_classes].argmax(-1, keepdim=True)
        best_scores = scores.max(dim=-1, keepdim=True)[0]
        converted_boxes = torch.cat((predicted_classes, best_scores, normalized_boxes), dim=-1)

        return converted_boxes.view(batch_size, grid_size * grid_size, -1)

    # Сохранение состояния модели
    def save_model_state(self, state, filename="model_checkpoint.pth"):
        torch.save(state, filename)

    # Загрузка состояния модели
    def load_model_state(self, checkpoint, model, optimizer):
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
