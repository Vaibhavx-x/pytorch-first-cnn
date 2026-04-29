import torch
import torchvision
import torchvision.transforms as transforms

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_train_loader(batch_size=128):
    train_transform = transforms.Compose([
        # Randomly flip the image horizontally 50% of the time
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Randomly shift the image by up to 10% (simulates objects not being perfectly centered)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        
        # Randomly rotate the image by up to 10 degrees
        transforms.RandomRotation(degrees=10),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))# Randomly erase a portion of the image 50% of the time, with a random size between 2% and 20% of the original image
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                       shuffle=True, num_workers=2)

def get_test_loader(batch_size=32):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    
    return torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                       shuffle=False, num_workers=2)