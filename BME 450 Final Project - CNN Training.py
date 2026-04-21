#BME 450 Spring 2026
#Final Project Using a Convolution Neural Network
#Written By: Nathan Petrucci, Mei Pitts
#Adapted from https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

#################################
### DELETE PREVIOUS TRAINING ####
#################################
import os
path = r"C:\Users\pitts\OneDrive - purdue.edu\Documents\09 - Semester 8\BME 450\00 - Final Project\trained_model.pt"
if os.path.exists(path):
    os.remove(path)

#################################
### IMPORT DATA AND TRANSFORM ###
#################################
resizeDim = 224 #The pixel number the images will be resized to

transform = transforms.Compose([        # Resize and normalize data
    transforms.Resize((resizeDim, resizeDim)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Import data for testing and training
batch_size = 16
if __name__ == "__main__":          # 
    trainset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Epic and CSCR hospital Dataset\\Train", transform=transform)
        # Data for entire Mendely Data
    # trainset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Train_Coronal", transform=transform)
        # Data for Coronal View only
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        # Loads data

    testset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Epic and CSCR hospital Dataset\\Test", transform=transform)
        # Data for entire Mendely Data
    # testset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Test_Coronal", transform=transform)
        # Data for Coronal View only
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        # Loads data

# Define the categories
classes = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# Functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#################################
##  CONVOLUTION NEURAL NETWORK ##
#################################
# Define the Net class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)     # Calculated based on pixel size of 224
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)     # 4 due to 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Resets network
net = Net()

#################################
########  TRAIN DATA  ###########
################################## 
# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9)

# Determine number of epochs and training loss/validation accuracy vectors
epochs = 10
train_losses = []
val_accuracies = []

for t in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Save average loss for this epoch for results graph
    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)

    # Compute validation accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Save validation accuracy for results graph
    val_accuracies.append(correct/total)

print('Finished Training! \n')

#################################
########  SAVE TRAINING  ########
#################################
torch.save(net.state_dict(), path)
print('Model Saved! \n')

#################################
####  TEST CASE WITH LABELS  ####
#################################
# print images
dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net.load_state_dict(torch.load(path, weights_only=True))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

#################################
#####  ACCURACY OF TRAINING #####
#################################
# Determine Accuracy of Training
# since we're not training, we don't need to calculate the gradients for our outputs
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n Accuracy of the network on the {total} test images: {100 * correct // total} % \n')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for {classname:5s} is {accuracy:.1f} %')

#################################
######## RESULTS GRAPH ##########
#################################
# Plot Training Loss and Validation Accuracy
epochs_range = range(1, len(train_losses) + 1)

# Plot Graph for Training Loss
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].plot(epochs_range, train_losses)
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Training Loss vs Epoch")
ax[0].grid(True)

# Plot Graph for Validation Accuracy
ax[1].plot(epochs_range, val_accuracies)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Validation Accuracy (%)")
ax[1].set_title("Validation Accuracy vs Epoch")
ax[1].grid(True)

plt.tight_layout()
plt.show()
