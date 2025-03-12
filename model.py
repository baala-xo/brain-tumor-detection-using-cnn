import torch
import torchvision.models as models
import os

def load_model(model_path="models/model.pth"):
    # ✅ Use absolute path for compatibility
    abs_model_path = os.path.join(os.path.dirname(__file__), model_path)

    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"❌ Model file not found at {abs_model_path}")

    # ✅ Load Pretrained ResNet18 but modify the classifier
    model = models.resnet18(pretrained=True)  # Load ImageNet weights

    # ✅ Modify Fully Connected (FC) Layer to match tumor classification (4 classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 4)  # Output: 4 classes
    )

    # ✅ Load trained model weights
    state_dict = torch.load(abs_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # ✅ Set to evaluation mode (stable predictions)
    model.eval()
    print("✅ Model loaded successfully!")
    
    return model
