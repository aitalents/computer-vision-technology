import torch 
import torch.nn as nn
from torchvision import models


class YOLOv1(nn.Module):
    def __init__(self, class_names, grid_size, img_size=(448,448)):
        super(YOLOv1,self).__init__()
        self.num_bbox = 2
        self.input_size = img_size
        self.class_names = class_names
        self.num_classes = len(class_names.keys())        
        self.grid = grid_size
        self.output_size = self.grid * self.grid * (self.num_bbox * 5 + self.num_classes)

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
