import torch
from torchvision import models
import torch.nn as nn
import cv2

PATH = 'models/resnet_True/Best_95.pth'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = models.resnet50(pretrained=True).to(device)
model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid()).to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

for i in range(1, 1209):
    image_np = cv2.imread('data/test/' + str(i) + '.jpg')
    image_torch = torch.from_numpy(image_np).unsqueeze(1).permute(1, 3, 0, 2).float()
    image_torch = image_torch.to(device) / 255.0
    out = model.forward(image_torch)
    if out.item() >= 0.5:
        print('George')
    else:
        print('Non-George')
    cv2.imshow('111', image_np)
    cv2.waitKey()
