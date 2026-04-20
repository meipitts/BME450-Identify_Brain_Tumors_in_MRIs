#BME 450 Spring 2026
#Working Final Project
#Written By: Nathan Petrucci, Mei Pitts
#Adapted from Homework 1 submission by Nathan Petrucci

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#Import and format training and test datasets
resizeDim = 256 #The pixel number the images will be resized to
image_transforms = transforms.Compose([
    transforms.Resize((resizeDim, resizeDim)),
    transforms.ToTensor(),
    #transforms.Normalize()
])

training_data = datasets.ImageFolder(root=r"C:\Users\pitts\OneDrive - purdue.edu\Documents\09 - Semester 8\BME 450\00 - Final Project\Train_Coronal", transform=image_transforms)
test_data = datasets.ImageFolder(root=r"C:\Users\pitts\OneDrive - purdue.edu\Documents\09 - Semester 8\BME 450\00 - Final Project\Test_Coronal", transform=image_transforms)

#Define the categories
categories = ['Glioma', 'Meningioma', 'Pituitary tumor', 'No tumor']

#Display a sample from the dataset
sample_num = 0
print('Inputs sample - image size:', training_data[sample_num][0].shape)
print('Label:', training_data[sample_num][1], '\n')
ima = training_data[sample_num][0]
#ima = (ima - ima.mean()) / ima.std()
iman = ima.permute(1, 2, 0)
plt.imshow(iman)
plt.show()


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
    
#Define the training and test loops
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

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
    accuracy = 100 * correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy

#Train the model
train_losses = []
model = Net()

batch_size = 1
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

val_accuracies = []
val_losses = []

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # train_loop(train_dataloader, model, loss_fn, optimizer)
    # test_loop(test_dataloader, model, loss_fn)
    
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    train_losses.append(train_loss)

    val_loss, val_acc = test_loop(test_dataloader, model, loss_fn)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Training Loss: {train_loss}")
    print(f"Validation Accuracy: {val_acc:.2f}%")    

print("Done!")

#Display Sample Test Data
samples = [2, 5, 7]
sampleCount = 0
with torch.no_grad():
    for X, y in test_dataloader:
        sampleCount += 1
        print(f'Sample Number {sampleCount}:')
        pred = model(X)
        print(f'--Expected Value {y}')
        print(f'--Actual Value: {pred}')

ima = test_data[sample_num][0]
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)
plt.show()

# Plot Training Loss Graph
epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid()
plt.show()

plt.plot(range(1, epochs + 1), val_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy vs Epoch")
plt.show()
plt.grid()

torch.save(model.state_dict(), r"C:\Users\pitts\OneDrive - purdue.edu\Documents\09 - Semester 8\BME 450\00 - Final Project\trained_model.pt")