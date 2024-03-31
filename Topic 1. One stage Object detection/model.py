import torch 
import torch.nn as nn
from torchvision import models

from utils import compute_pairwise_iou


class YOLOv1(nn.Module):
    def __init__(
        self, class_names, grid_size,
        img_size=(448,448), device='cpu',
    ):
        super(YOLOv1,self).__init__()
        self.num_bbox = 2
        self.input_size = img_size
        self.class_names = class_names
        self.num_classes = len(class_names.keys())        
        self.grid = grid_size
        self.output_size = self.grid * self.grid * (self.num_bbox * 5 + self.num_classes)

        self.device = device

        self.eps = 1e-8 # for loss computation: small addition before get sqrt of w and h

        # backbone = feature extractor
        resnet50 = models.resnet50(pretrained=True)
        self.extraction_layers = nn.Sequential(*list(resnet50.children())[:-2])
        self.neck_in = 2048 # for resnet50

        # neck: refining higher level semantic information
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.neck_in, 1024, 3, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(p=0.5),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((7,7))
        )
        
        # head: make final predictions with shape (batch_size, grid_size, grid_size, B * 5 + C)
        self.linear_layers = nn.Sequential(
            nn.Linear(50176, 12544, bias=False),
            nn.BatchNorm1d(12544),
            nn.Dropout(p=0.1), 
            nn.LeakyReLU(0.1),

            nn.Linear(12544, 3136, bias=False),
            nn.BatchNorm1d(3136),
            nn.LeakyReLU(0.1),

            nn.Linear(3136, self.output_size, bias=False),
            nn.BatchNorm1d(self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]

        actvs = self.extraction_layers(x)
        actvs = self.final_conv(actvs)

        lin_input = torch.flatten(actvs)(batch_size, -1)
        lin_out = self.linear_layers(lin_input)
        det_tensor = lin_out.view(-1, self.grid, self.grid, (self.num_bbox * 5) + self.num_classes)

        return det_tensor


    def get_best_bboxes_from_predicted(self, pred, target):
        """
            Input:
                pred -- model output for one image;
                        has shape (grid_size, grid_size, num_bbox * 5 + num_classes)
                target -- yolo target for one image;
                        has shape (grid_size, grid_size, 5)
            Output:
                best_bboxes: tensor with shape (grid_size, grid_size, 5 + num_classes),
                            that indicates what pred_bboxes use in Loss computation:
                            1. for cells with gt_bbox center choose pred_bbox with the highest IoU
                            2. for cells without gt_bbox center choose pred_bbox with the highest confidence (p_c)
        """
        pred_bboxes = pred[:, :, :self.num_bbox * 5]
        pred_classes = pred[:, :, self.num_bbox * 5:]

        best_bboxes = torch.zeros((self.grid, self.grid, 5 + self.num_classes))
        best_bboxes[:, :, 5:] = pred_classes

        for i in range(self.grid):
            for j in range(self.grid):
                if target[i, j, :].sum() > 0:
                    ious = compute_pairwise_iou(
                        torch.reshape(pred_bboxes[i, j, :], (-1, 5)),
                        torch.reshape(target[i, j, :], (1, 5)),
                    )
                    bbox_idx = torch.argmax(ious)
                else:
                    confidences = pred_bboxes[i, j, 4::5]
                    bbox_idx = torch.argmax(confidences)
                
                best_bboxes[i, j, :] = pred_bboxes[i, j, 5 * bbox_idx : 5 * (bbox_idx + 1)]

        return best_bboxes


    def compute_loss(
        self, model_output, target,
        lambda_coord=5, lambda_noobj=0.5,
    ):
        batch_loss = torch.tensor(0).float().to(self.device)

        for i, pred in enumerate(model_output):
            item_loss = torch.tensor(0).float().to(self.device)
            best_bboxes = self.get_best_bboxes_from_predicted(pred, target[i])

            pred_classes = best_bboxes[:, :, 5:]
            gt_classes = torch.zeros_like(pred_classes)

            for j in range(self.grid):
                for k in range(self.grid):
                    gt_classes[j, k, target[i, j, k, 4]] = 1

                    if target[i, j, k, :].sum() > 0: # case: there is gt bbox center in cell 
                        cell_loss = lambda_coord * torch.pow(best_bboxes[j, k, :2] - target[i, j, k, :2], 2).sum() + \
                            + lambda_coord * torch.pow(torch.sqrt(best_bboxes[j, k, 2:4] + self.eps) - torch.sqrt(target[i, j, k, 2:4] + self.eps), 2).sum() + \
                            + torch.pow(best_bboxes[j, k, 4] - 1, 2) + \
                            + torch.pow(pred_classes[j, k, :] - gt_classes[j, k, :], 2).sum()
                    else:
                        # confidence should be zero
                        cell_loss = lambda_noobj * torch.pow(best_bboxes[j, k, 4] - 0, 2)

                    item_loss += cell_loss

            batch_loss += item_loss

        mean_batch_loss = batch_loss / model_output.size()[0]

        return mean_batch_loss
