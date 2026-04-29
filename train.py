import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_test_loader, get_train_loader
from resnet import Net

def main():
    # 1. DEFINE THE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # 2. CREATE AN INSTANCE AND MOVE IT
    net = Net().to(device) 

    # 3. SETUP DATA LOADER
    train_loader = get_train_loader(batch_size=128)
    testloader = get_test_loader(batch_size=32)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Use label smoothing to help prevent overfitting
    # Pass the instance parameters, not the class
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=False) # Use SGD with momentum, weight decay for regularization, and Nesterov momentum
    # optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4) # Use Adam optimizer with weight decay for regularization    
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    print(f'Using optimizer: {optimizer.__class__.__name__}')
    print(f'Using scheduler: {scheduler.__class__.__name__}')


    best_acc = 0
    patience = 25
    trigger_times = 0
    scheduler_offset = 0  # This will track how many epochs have passed since the last scheduler reset

    # 1. Initialize the Scaler before the loop
    scaler = torch.amp.GradScaler('cuda') 

    for epoch in range(500):
        running_loss = 0.0
        net.train()  # Set the model to training mode

        # if epoch == 26:
        #     print('Updating optimizer: Changing max LR to 0.001 and weight decay to 1e-4...')
            
        #     # 1. Safely update the existing optimizer parameters without losing momentum
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.001          # Update LR so the new scheduler registers it
        #         param_group['initial_lr'] = 0.001    # Update initial LR for the scheduler
        #         param_group['weight_decay'] = 1e-4 # Update weight decay
            
        #     # 2. Recreate the scheduler so it picks up the new 0.001 base learning rate
        #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
            
        #     # 3. Set the offset to 26 so the new scheduler thinks it's starting at epoch 0
        #     scheduler_offset = 26
        
        
        for i, data in enumerate(train_loader, 0):
            # 4. MOVE DATA TO GPU
            # data contains [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()# Zero the parameter gradients

            # 5. FORWARD PASS ON GPU
            # 2. Wrap the forward pass and loss in autocast
            with torch.amp.autocast('cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()# Update the weights
            # Update the learning rate based on the epoch and batch index for CosineAnnealingWarmRestarts scheduler
            # scheduler.step(epoch - scheduler_offset + i / len(train_loader))  
            
            running_loss += loss.item()
            if i % 50 == 49:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0
        
        net.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0
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

        
        accuracy = 100 * correct / total
        print(f'Accuracy of the network on {epoch + 1} epoch: {accuracy} %')
        
        if accuracy > best_acc:
            best_acc = accuracy
            print(f'New best accuracy: {best_acc:.2f}%. Saving model...')
            trigger_times = 0
            #save the model
            torch.save(net.state_dict(), 'checkpoints/cifar_net.pth')
        else:
            trigger_times += 1
            print(f'EarlyStopping counter: {trigger_times} out of {patience}')
            print(f'Current best accuracy: {best_acc:.2f}%')
            if trigger_times >= patience:
                print(f'Early stopping at {epoch + 1}!')
                print(f'Best accuracy: {best_acc:.2f}%')
                break
        
        # scheduler.step() # use this for StepLR scheduler
        scheduler.step(accuracy) # use this for ReduceLROnPlateau scheduler
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current learning rate: {current_lr:.5f}')

            

    print('Finished Training')


if __name__ == '__main__':
    main()