#BME 450 Spring 2026
#Trained Neural Net Loader
#Written By: Nathan Petrucci, Mei Pitts

#Loads in the saved .pt file produced after running WorkingProjectCode and runs the test dataset with it

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

resizeDim = 128 #The pixel number the images will be resized to
image_transforms = transforms.Compose([
    transforms.Resize((resizeDim, resizeDim)),
    transforms.ToTensor(),
    #transforms.Normalize()
])

test_data = datasets.ImageFolder(root=r"C:\Users\npetr\Downloads\Brain Tumor MRI Dataset (Glioma, Meningioma, Pitui\Brain Tumor MRI Dataset (Glioma, Meningioma, Pitui\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Test_Abridged_1", transform=image_transforms)

categories = ['Glioma', 'Meningioma', 'Pituitary tumor', 'No tumor']

#Define the Net class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(resizeDim*resizeDim*3, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 4)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = self.l3(x)
        return output
    
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

model = Net()
model.load_state_dict(torch.load(r"C:\Users\npetr\Downloads\BME450_Project_Datasets\Attempt 3\Attempt3.pt", weights_only=True))
model.eval()


batch_size = 1
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
loss_fn = nn.CrossEntropyLoss()

test_loop(test_dataloader, model, loss_fn)

sampleCount = 0
with torch.no_grad():
    for X, y in test_dataloader:
        sampleCount += 1
        print(f'Sample Number {sampleCount}:')
        pred = model(X)
        print(f'--Expected Value {y}')
        print(f'--Actual Value: {pred}')