import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F # Bunu en üste import kısmına eklemeyi unutma!

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) 
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) 
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) 
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 16 * 16, out_features=512)
        self.bn4 = nn.BatchNorm1d(512) 
        self.relu4 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=512, out_features=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.bn4(self.fc1(x))))
        x = self.fc2(x) 
        return x

class EfficientNetDeepfake(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EfficientNetDeepfake, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.efficientnet(x)

# YENİ EKLENEN RESNET18 MODELİ
class ResNet18Deepfake(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ResNet18Deepfake, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.resnet(x)
    

# YENİ EKLENEN VİSİON TRANSFORMER (ViT) MODELİ
class ViTDeepfake(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ViTDeepfake, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # ViT zorunlu olarak 224x224 resim ister.
        # Sistemdeki 128x128 resimleri bozmamak için modelin içinde anlık büyütüyoruz:
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vit(x)