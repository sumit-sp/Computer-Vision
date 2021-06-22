## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 3)
        # ==> 32x222x222
        
        ## Note that among the layers to add, consider including:
        # maxpooling, multiple conv, fully-connected, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.pool = nn.MaxPool2d(2, 2)
        # ==> 32x111x111
        
        # second convolution layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        # ==> 64x109x109
        
        # after 2nd maxpooling ==> 64x54x54
        
        #3rd convolution layer
        self.conv3 = nn.Conv2d(64, 128, 3)
        # ==> 128x52x52
        # after maxpooling ==>128x26x26
        
        # Linear layer ==> We need 136 keypoints (x,y co-ordinates)
        self.fc1 = nn.Linear(128*26*26, 1024)

        # dropout with p=0.3
        self.fc1_drop = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(1024, 512)
        
        self.fc2_drop = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(512, 68*2)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.fc1_drop(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.fc1_drop(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        
        #Preparation for linear layaer ==> flatten
        x = x.view(x.size(0),-1)
        
        #Linear layer with Dropout
        x = F.relu(self.fc1(x))
        
        x = self.fc1_drop(x)
        
        x = F.relu(self.fc2(x))
        
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        
        # To check whether output size is correct
        #print(x.size())
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        return x

