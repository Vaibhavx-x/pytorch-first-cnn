import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_test_loader,classes
from resnet import Net

# Load the trained model
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.load_state_dict(torch.load('checkpoints/cifar_net.pth'))
    net.to(device)  # Move the model to the GPU if available
    net.eval()  # Set the model to evaluation mode

    testloader = get_test_loader(batch_size=32)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move data to the GPU if available
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == '__main__':
    main()