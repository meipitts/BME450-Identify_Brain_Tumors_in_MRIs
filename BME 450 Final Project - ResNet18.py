# BME 450 Spring 2026
# Final Project Using a ResNet18 Convolution Neural Network
# Written By: Nathan Petrucci, Mei Pitts
# Adapted from https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

#################################
######## IMPORT PACKAGES ########
#################################
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
from torchvision.models import resnet18

if __name__ == "__main__":           
    #################################
    ##### TOTAL CODE RUN TIME  ######
    #################################
    import time
    start = time.time()
    print("Timer Started")
    
    #################################
    ### DEVICE AND SPEED SETTINGS ###
    #################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
    torch.backends.cudnn.benchmark = True
    resizeDim = 128 #The pixel number the images will be resized to

    # GPU transforms if available
    try:
        from torchvision.transforms.v2 import Resize, ToTensor, Normalize
        transform = transforms.Compose([
            Resize((resizeDim, resizeDim)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])
    except:
        # fallback for older torchvision
        transform = transforms.Compose([
            transforms.Resize((resizeDim, resizeDim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Import data for testing and training
    batch_size = 64     # Choosen batch size  to optimze speed of code

    # Titles based on dataset used  
    dataset_type = 'Mendely Data'
    # dataset_type = 'Coronal View Data'
    
    # Import and load training data set
    trainset = datasets.ImageFolder(root="C:\Epic and CSCR hospital Dataset\Train", transform=transform)
        # Data for entire Mendely Data
    # trainset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Train_Coronal", transform=transform)
        # Data for Coronal View only
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # Loads data

    # Import and load testing data set
    testset = datasets.ImageFolder(root="C:\Epic and CSCR hospital Dataset\Test", transform=transform)
        # Data for entire Mendely Data
    # testset = datasets.ImageFolder(root="C:\\Users\\pitts\\OneDrive - purdue.edu\\Documents\\09 - Semester 8\\BME 450\\00 - Final Project\\Test_Coronal", transform=transform)
        # Data for Coronal View only
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # Loads data

    # Define the categories
    classes = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']     # Categories given from Mendely Data

    #################################
    ##  CONVOLUTION NEURAL NETWORK ##
    #################################
    # Define the Net class (ResNet18 CNN)
    net = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    for param in net.parameters():
        param.requires_grad = False

    net.fc = nn.Linear(512, 4)
    net = net.to(device)

    #################################
    ########  TRAIN DATA  ###########
    ################################## 
    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.fc.parameters(), lr=1e-4)

    # Determine number of epochs and training loss/validation accuracy vectors
    epochs = 50         # Larger # of epochs for more training loops
    train_losses = []   
    val_accuracies = []

    # Indicates start of training sequence   
    print("Starting Training")

    for t in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0  # Define to later use for training loss
        
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)

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
        if t % 2 == 0:
            net.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            val_accuracies.append(val_acc)
            
            # Print statement for every set of epochs finished (used to monitor code running efficiently)
            print(f"Epoch {t+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.3f}")

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
    images_gpu = images.to(device)

    net.load_state_dict(torch.load(path))
    net = net.to(device)

    outputs = net(images_gpu)
    _, predicted = torch.max(outputs, 1)

    # Unnormalize function
    def unnormalize(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.numpy().transpose((1, 2, 0))
        img = std * img + mean
        return np.clip(img, 0, 1)

    # Plot images with labels
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for idx in range(4):
        
        # Define border colors (green = correct, red = incorrect)
        correct = (predicted[idx] == labels[idx])
        border_color = 'green' if correct else 'red'

        # Display original image and class name
        ax_top = axes[0, idx]
        ax_top.imshow(unnormalize(images[idx]))
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(f"Original Class: {classes[labels[idx]]}", fontsize=11)

        #  Add border to original image based on if guess image is correct or not
        for spine in ax_top.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        # Display guessed image and class name based on training
        ax_bottom = axes[1, idx]
        ax_bottom.imshow(unnormalize(images[idx]))
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_bottom.set_title(f"Predicted Class: {classes[predicted[idx]]}", fontsize=11)

        # Add border to original image based on if guess image is correct or not
        for spine in ax_bottom.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    # Overall title and display images
    fig.suptitle(f'Sample Results for ResNet18 CNN and {dataset_type}', fontsize=16)
    plt.tight_layout()
    plt.show()

    #################################
    #####  ACCURACY OF TRAINING #####
    #################################
    # Determine Accuracy of Training
    correct = 0
    total = 0

    ## OVERALL ACCURACY OF TEST IMAGES ##
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nAccuracy of the network on the {total} test images: {100 * correct // total} % \n')

    ## ACCURACY OF TEST IMAGES BROKEN DOWN BY CLASS ##
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for {classname:5s} is {accuracy:.1f} %')

    print('\n')

    #################################
    ######## RESULTS GRAPH ##########
    #################################
    # Plot Training Loss and Validation Accuracy
    epochs_range = range(1, len(train_losses) + 1)
    val_epochs = list(range(1, epochs + 1, 2))     # Needed since validation does not happen every epoch

    # Plot Results Graphs
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    # Plot Graph for Training Loss
    ax[0].plot(epochs_range, train_losses)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss vs Epoch")
    ax[0].grid(True)

    # Plot Graph for Validation Accuracy
    ax[1].plot(val_epochs, val_accuracies)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Validation Accuracy (%)")
    ax[1].set_title("Validation Accuracy vs Epoch")
    ax[1].grid(True)

    # Overall title and display images
    fig.suptitle(f'Results Graphs for {len(epochs_range)} Epochs and {total} Images', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the title
    plt.show()

    #################################
    ##### TOTAL CODE RUN TIME  ######
    #################################
    end = time.time()
    print(f"Total runtime of the program is {end - start} seconds")