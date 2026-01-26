import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoFeatureExtractor, SwinForImageClassification

from urllib.request import urlopen
from PIL import Image
import timm

def ResNet_50(in_channels=3, num_classes=2):
    """
    Creates a ResNet-50 model pretrained on ImageNet1k..
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        num_classes (int): Number of output classes
    
    Returns:
        model (torch.nn.Module): Adapated to ResNet-50 224 Pre trained model
        transform (torchvision.transforms): Tranformation adapted for 384x384 image resolution
    """
    
    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained ResNet-50")

    # load model
    weights = 'IMAGENET1K_V1'
    model = models.resnet50(weights=weights)

    # get the input features and change the head. Preping for transfer learning
    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_ftrs, num_classes, bias=True)

    transform = weights.transforms()          # default: crop_size = [224], resize_size[256]
    '''
    DEFAULT transform
    ImageClassification(
    crop_size=[224]
    resize_size=[256]
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    interpolation=InterpolationMode.BILINEAR
    )
    '''
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    
    transform384 = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
    ])

    return model, transform384

def ResNet18_224(in_channels=3, num_classes=2):
    """
    Creates a ResNet-18 model pretrained on ImageNet1k.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        num_classes (int): Number of output classes
    
    Returns:
        model (torch.nn.Module): Adapated to ResNet-18 224 Pre trained model
        transform (torchvision.transforms): Tranformation for 224x224 image resolution
    """
    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained ResNet-50")

    # load model
    model = models.resnet18(weights='IMAGENET1K_V1')

    # get the input features and change the head. Prep for transfer learning
    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_ftrs, num_classes, bias=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = {"train": T.Compose([
            T.Resize((224, 224)),

            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),

            T.ToTensor(),
        ]),

        "val": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])}

    return model, transform

def ResNet_50_224(in_channels=3, num_classes=2, pre_trained=True):

    if pre_trained:
        model = models.resnet50('IMAGENET1K_V1')
    else:
        model = models.resnet50()

    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained ResNet-50")
    
    in_ftrs = model.fc.in_features
    # model.fc = nn.Linear(in_ftrs, num_classes, bias=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_ftrs, num_classes)
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = {
        "train": T.Compose([
            T.Resize((224, 224)),
            
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ]),
        "val": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    }
    
    return model, transform
    

def EfficientNet(in_channels=3, num_classes=2, pre_trained=True):
    """
    Creates an EfficientNetB4 model pretrained on ImageNet1k.
    
    Args:
        in_channels (int): Number of input channels (3=RGB, 1=grayscale)
        num_classes (int): Number of output classes
    
    Returns:
        model (torch.nn.Module): EfficientNetV2-L pretrained model adapted for num_classes
        transform (callable): Timm-compatible validation transform for 384x384 images
    """

    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained EfficientNet.")

    model = models.efficientnet_b4(weights='IMAGENET1K_V1')

    in_ftrs = model.classifier[1].in_features
    model.classifier = nn.Linear(in_ftrs, num_classes, bias=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = {"train": T.Compose([
            T.Resize((224, 224)),

            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),

            T.ToTensor(),
            ]),

            "val": T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])}

    return model, transform

def Swin_B(in_channels=3, num_classes=2, pre_trained=True):
    # got this from https://huggingface.co/microsoft/swin-base-patch4-window12-384
    """
    Creates a Swin-Base model pretrained on ImageNet1k.
    
    Args:
        in_channels (int): Input channels (3=RGB)
        num_classes (int): Number of output classes
    
    Returns:
        model (torch.nn.Module): Pretrained Swin-Base model adapted for num_classes
        transform (torchvision.transforms.Compose): Preprocessing transform for 384x384 input
    """
    
    if in_channels != 3:
        raise ValueError("Input channels must be 3 for pretrained Swin-Base.")

    # load pretrained model
    model = models.swin_b(weights='IMAGENET1K_V1')
    
    # adapt classifier to your number of classes
    in_ftrs = model.head.in_features
    model.head = nn.Linear(in_ftrs, num_classes, bias=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = {"train": T.Compose([
            T.Resize((224, 224)),

            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),

            T.ToTensor(),
            ]),

            "val": T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])}

    return model, transform

class _ConvBase(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
            self.flatten_dim = out.numel()

        # Classifier (MLP)
        self.MLP = nn.Linear(self.flatten_dim, num_classes)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        return x

class _ConvTiny(nn.Module):
    def __init__(self, in_channels, num_classes, input_size):
        super().__init__()

        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
            self.flatten_dim = out.numel()

        # Classifier (MLP)
        self.MLP = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.MLP(x)
        return x

def get_ConvBase(in_channels, num_classes, pre_trained=True):

    model = _ConvBase(in_channels, num_classes, input_size=224)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = {"train": T.Compose([
            T.Resize((224, 224)),

            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),

            T.ToTensor(),
        ]),

        "val": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])}
    
    return model, transform

def get_ConvTiny(in_channels, num_classes, pre_trained=True):

    model = _ConvTiny(in_channels, num_classes, input_size=224)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = {"train": T.Compose([
            T.Resize((224, 224)),

            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),

            T.ToTensor(),
        ]),

        "val": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])}
    
    return model, transform
    
  
