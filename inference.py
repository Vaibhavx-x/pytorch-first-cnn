import torch
import torchvision.transforms as transforms
from PIL import Image
from resnet import Net  # Make sure this matches the file where your Net class is

# The 10 categories your model knows
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build the brain (Load architecture)
    net = Net().to(device)

    # 2. Load the memories (Your 90% accurate weights)
    # Using weights_only=True is a modern PyTorch security standard
    try:
        net.load_state_dict(torch.load('checkpoints/cifar_net.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: Could not find 'checkpoints/cifar_net.pth'. Did you save it somewhere else?")
        return
        
    net.eval()  # Lock it into testing mode

    # 3. Define the preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Shrink it to CIFAR size
        transforms.ToTensor(),
        # Use the exact same normalization from your training script!
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 4. Open the image
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Could not find your image at '{image_path}'")
        return

    # Apply transforms and add the batch dimension: (C, H, W) -> (1, C, H, W)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # 5. Make the prediction
    with torch.no_grad():
        outputs = net(batch_t)
        
        # Softmax turns the raw numbers into percentage probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # 6. Display the results!
    print(f"\n--- AI Vision Results for '{image_path}' ---")
    
    # Get the top 3 guesses
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    for i in range(3):
        predicted_class = classes[top3_idx[i]]
        confidence = top3_prob[i].item() * 100
        
        if i == 0:
            print(f"Top Guess: {predicted_class.upper()} ({confidence:.2f}%)")
        elif i == 1:
            print(f"2nd Guess: {predicted_class.capitalize()} ({confidence:.2f}%)")
        else:
            print(f"3rd Guess: {predicted_class.capitalize()} ({confidence:.2f}%)\n")

if __name__ == '__main__':
    # REPLACE THIS with the name of a picture you saved to your folder!
    target_image = 'my_test_image.jpg'
    predict_image(target_image)