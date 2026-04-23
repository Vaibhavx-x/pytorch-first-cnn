import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_train_loader
from model import Net

def main():
    # 1. DEFINE THE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # 2. CREATE AN INSTANCE AND MOVE IT
    net = Net().to(device) 

    # 3. SETUP DATA LOADER
    train_loader = get_train_loader(batch_size=64)

    criterion = nn.CrossEntropyLoss()
    # Pass the instance parameters, not the class
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # 4. MOVE DATA TO GPU
            # data contains [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            # 5. FORWARD PASS ON GPU
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # 6. SAVE FOR LATER
    torch.save(net.state_dict(), 'checkpoints/cifar_net.pth')

if __name__ == '__main__':
    main()