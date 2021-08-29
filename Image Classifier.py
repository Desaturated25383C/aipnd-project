# Imports here
%matplotlib inline
%config InlineBackend.figure_format = "retina"

import matplotlib.pyplot as plt 

import torch
from torch import nn, optim

import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import io
import pathlib

from workspace_utils import active_session

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Pretrained network loaded from torchvision.models
model = models.vgg16(pretrained=True)

## Freeze parameters
for param in model.parameters():
    param.require_grad = False
    
model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 #nn.Linear(4096, 4096),
                                 #nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

# TODO: Do validation on the test set
with active_session():
    epochs = 4
    steps = 0
    
    train_losses, test_losses = [], []
    
    for e in range(epochs):
        running_loss = 0
        
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        else:
            test_loss = 0
            accuracy = 0
            
            # Turn off gradients for validation
            with torch.no_grad():
                
                # set model to evaluation mode 
                model.eval()
                
                # validation pass
                for images, labels in testloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images)
                    test_loss += criterion(logps, labels)
                    
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            # set model back to train mode
            model.train()
            
            
# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.class_to_idx()}


torch.save(checkpoint, model.class_to_idx)
