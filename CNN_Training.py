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

import os

path = r"C:\Users\pitts\OneDrive - purdue.edu\Documents\09 - Semester 8\BME 450\00 - Final Project\trained_model.pt"
if os.path.exists(path):
    os.remove(path)

#Import and format training and test datasets

resizeDim = 224 #The pixel number the images will be resized to

transform = transforms.Compose([
    transforms.Resize((resizeDim, resizeDim)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4

if __name__ == "__main__":
    trainset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Epic and CSCR hospital Dataset\\Train", transform=transform)
    # trainset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Train_Coronal", transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Epic and CSCR hospital Dataset\\Test", transform=transform)
    # testset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Test_Coronal", transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

#Define the categories
classes = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#Define the Net class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
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

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9)

epochs = 3
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

    # Save average loss for this epoch
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

    val_acc = correct / total
    val_accuracies.append(val_acc)

print('Finished Training! \n')

torch.save(net.state_dict(), path)
print('Model Saved! \n')

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# net = Net()
net.load_state_dict(torch.load(path, weights_only=True))
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# Determine Accuracy of Training
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

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


# Plot Training Loss and Validation Accuracy (Not working)
epochs = range(1, len(train_losses) + 1)

# plt.figure(figsize=(10,5))

# Plot training loss
# plt.subplot(1, 2, 1)
# plt.plot(epochs, train_losses, marker='o')
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(epochs, train_losses)
ax[0].set_ylim(0, 2)
ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")

# Plot validation accuracy
# plt.subplot(1, 2, 2)
# plt.plot(epochs, val_accuracies, marker='o', color='green')
ax[1].plot(epochs, val_accuracies)
ax[1].set_title("Validation Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()