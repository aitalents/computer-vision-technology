
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import efficientnet_v2_s, mobilenet_v3_large


def load_efficientnet_v2_s(num_classes=1000, freeze_weights=True):
    # Load pre-trained model
    model = efficientnet_v2_s(weights='IMAGENET1K_V1', num_classes=num_classes)

    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    # Remove the top layer (classification head)
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Identity()
    return model, features


def load_mobilenet_v3_l(num_classes=1000, freeze_weights=False):
    # Load pre-trained model
    model = mobilenet_v3_large(weights='IMAGENET1K_V1', num_classes=num_classes)

    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier = nn.Identity()

    features = 960  # This is based on the last channel size of MobileNetV3 before the classifier.
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    return model, features

# Implement the YOLO v1 detection head
class YOLOv1Head(nn.Module):
    def __init__(self, features, S=7, B=2, C=80):
        super(YOLOv1Head, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.output_dim = S * S * (C + B * 5)  # 5 = x, y, w, h, confidence

        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, self.output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.detector[:-1]:  # Apply to all but last layer
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.detector(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x


# Combine EfficientNet V2 S and YOLO v1 head
class EfficientNetV2S_YOLOv1(nn.Module):
    def __init__(self):
        super(EfficientNetV2S_YOLOv1, self).__init__()
        self.backbone, features = load_mobilenet_v3_l()
        self.yolo_head = YOLOv1Head(features)

    def forward(self, x):
        x = self.backbone(x)
        x = self.yolo_head(x)
        return x
