import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Load the pre-trained ResNet18 model
        # 'DEFAULT' means load the best available pre-trained weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Replace the giant 7x7 filter with a standard 3x3 filter that doesn't shrink the image
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 3. Delete the MaxPool layer entirely (replace it with an Identity layer that does nothing)
        self.resnet.maxpool = nn.Identity()
        
        # 4. Modify the final layer
        # The original ResNet18 was trained to recognize 1,000 different things.
        # We need to change the final Linear layer to output exactly 10 classes.
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet(x)