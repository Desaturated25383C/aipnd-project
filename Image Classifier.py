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
              'classifier' : model.classifier,
              'state_dict': model.class_to_idx}


torch.save(checkpoint,'checkpoint.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['classifier'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
  
  
 # TODO: Process a PIL image for use in a PyTorch model 
from PIL import Image
import glob, os

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((
        size[0]//2 - 112,
        size[1]//2 - 112,
        size[0]//2 + 112,
        size[1]//2 + 112)
    )
    np_image = np.array(image)
    #Scale Image per channel
    # Using (image-min)/(max-min)
    np_image = np_image/255.
        
    img_a = np_image[:,:,0]
    img_b = np_image[:,:,1]
    img_c = np_image[:,:,2]
    
    # Normalize image per channel
    img_a = (img_a - 0.485)/(0.229) 
    img_b = (img_b - 0.456)/(0.224)
    img_c = (img_c - 0.406)/(0.225)
        
    np_image[:,:,0] = img_a
    np_image[:,:,1] = img_b
    np_image[:,:,2] = img_c
    
    # Transpose image
    np_image = np.transpose(np_image, (2,0,1))
    return np_image
  
  def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
  
  def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Use the GPU if its available
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to('cpu')
    
    #Switch the model to evaluation mode to turn off dropout
    model.eval()
    
    with torch.no_grad():
    # Implement the code to predict the class from an image file    
        # Processs the image
        image = Image.open(image_path)
        image = process_image(image)

        # We need a tensor for the model so change the image to a np.Array and then a tensor
        image = torch.from_numpy(np.array([image])).float()
        image.to('cpu')

        # Use the model to make a prediction
        logps = model(image)
        ps = torch.exp(logps)

        # Get the top 5 probabilities and classes. This is returned as a tenosr of lists
        p, classes = ps.topk(topk, dim=1)

        # Switch the model back to training mode
        #model.train()
        
        # Get the first items in the tensor list to get the list of probs and classes
        top_p = p.tolist()[0]
        top_classes = classes.tolist()[0]
        
        # Reverse the categories dictionary
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        
        # Get the lables from the json file
        labels = []
        for c in top_classes:
            labels.append(cat_to_name[idx_to_class[c]])
    
        return top_p, labels
    
    # TODO: Display an image along with the top 5 classes
# Display an image along with the top 5 classes
# Create a plot that will have the image and the bar graph
fig = plt.figure(figsize = [10,5])

# Create the axes for the flower image 
ax = fig.add_axes([.5, .4, .5, .5])

# Process the image and show it
result = process_image('flowers/test/77/image_00005.jpg')
ax = imshow(result, ax)
ax.axis('off')
index = 77

ax.set_title(cat_to_name[str(index)])


# Make a prediction on the image
predictions, classes = predict('flowers/test/77/image_00005.jpg', model)


# Make a bar graph
# Create the axes for the bar graph
ax1 = fig.add_axes([0, -.5, .775, .775])

# Get the range for the probabilities
y_pos = np.arange(len(classes))

# Plot as a horizontal bar graph
plt.barh(y_pos, predictions, align='center', alpha=0.5)
plt.yticks(y_pos, classes)
plt.xlabel('probabilities')
plt.show()
