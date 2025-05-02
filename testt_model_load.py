import torch

try:
    model = torch.load("model.pth", map_location="cpu")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
