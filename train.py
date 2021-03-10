import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import Dataset, ShowImage

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels):
        for t in self.transforms:
            img, labels = t(img), labels
        return img, labels

learning_rate = 1e-5
epochs = 1500
batch_size = 12
# How often algorithm will calculate Train Accuracy and Test Accuracy. That's necessary to speed up learning
epoch_accuracy = 1
TRAIN_IMAGE_DIR = 'data/train'
TEST_IMAGE_DIR = 'data/test'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Transfer Learning Pretrained Resnet-50 from PyTorch
model = models.resnet50(pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False
# Add Classification FC layer with 1 output neuron
model.fc = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid()).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# print(model)
# Loss Function - Binary Cross-Entropy
loss_fn = nn.BCELoss()
model.train()

# Form Dataset
    # Train Dataset
transform = Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = Dataset(csv_file='train.csv', root_dir=TRAIN_IMAGE_DIR, transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          num_workers=0,
                          pin_memory=True, shuffle=True,
                          drop_last=True)
    # Test Dataset
test_dataset = Dataset(csv_file='test.csv', root_dir=TEST_IMAGE_DIR, transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          num_workers=0,
                          pin_memory=True, shuffle=True,
                          drop_last=True)

# Check Images in Train Dataset
for x, y in train_loader:
    # si = ShowImage(x, y)
    # si.show_batch()
    None

best_test_pth = -1
with open('log.txt', 'a') as log:
    log.write('Begin \n')
for epoch in range(epochs):
    mean_loss = []
    print(f'********* Epoch № {epoch} *********')
    loop = tqdm(train_loader, leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        y = y.unsqueeze(1)
        x, y = x.to(device), y.to(device)
        out = model.forward(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    print(f'Mean loss: {sum(mean_loss) / len(mean_loss)}')
    # Calculate the Accuracy on Train Dataset
    if epoch % epoch_accuracy == 0:
        model.eval()
        true_classifier = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            y = y.unsqueeze(1)
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            # print(f'Predicted Value: {out}')
            # print(f'Ground truth Value: {y}')
            for i in range(0, batch_size):
                if (out[i].item() >= 0.5 and y[i].item() == 1.0) or (out[i].item() < 0.5 and y[i].item() == 0.0):
                    true_classifier += 1
        model.train()
        accuracy_train = true_classifier / (len(train_loader.dataset))
        print(f'Train Accuracy: {accuracy_train}')

    # Calculate the Accuracy on Test Dataset
    if epoch % epoch_accuracy == 0:
        model.eval()
        true_classifier = 0
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            out = out.view(1)
            # print(f'Predicted Value: {out}')
            # print(f'Ground truth Value: {y}')
            if (out.item() >= 0.5 and y.item() == 1.0) or (out.item() < 0.5 and y.item() == 0.0):
                true_classifier += 1
        model.train()
        accuracy = true_classifier / (batch_idx + 1)
        if accuracy > best_test_pth:
            best_test_pth = accuracy
            print('Save Model')
            torch.save(model.state_dict(), 'Best.pth')
        print(f'Test Accuracy: {accuracy}')
    if epoch % epoch_accuracy == 0:
        with open('log.txt', 'a') as log:
            log.write('*** Epoch # ' + str(epoch) + ' ***\n')
        with open('log.txt', 'a') as log:
            log.write('Mean Loss: ' + str(sum(mean_loss) / len(mean_loss)) + '\n')
        with open('log.txt', 'a') as log:
            log.write('Train Accuracy: ' + str(accuracy_train) + '\n')
        with open('log.txt', 'a') as log:
            log.write('Test Accuracy: ' + str(accuracy) + '\n')
    if epoch % epoch_accuracy == 0:
        torch.save(model.state_dict(), str(epoch) + '.pth')

print('Save Model')
torch.save(model.state_dict(), 'Full.pth')
