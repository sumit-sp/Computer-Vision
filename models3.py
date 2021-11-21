import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
      
        ## input image size is 224 * 224 pixels        
        # first convolutional layer
        ## (W-F)/S + 1 = (224-5)/1 + 1 = 220
        ## self.conv1 = nn.Conv2d(1, 32, 3) # the output Tensor for one image, will have the dimensions: (32, 222, 222)

        self.conv1 = torch.nn.Conv2d(1,20,3)
        #==> 20x111x111
        
        self.conv2 = torch.nn.Conv2d(20, 30, 3)
        #==> 30x54x54
        
        self.conv3 = torch.nn.Conv2d(30, 40, 3)
        #==> 40x26x26
        
        self.conv4 = torch.nn.Conv2d(40, 60, 3)
        #==> 60x12x12
        
        self.conv5 = torch.nn.Conv2d(60, 100, 3)
        #==> 100x5x5
        
        # Max-pooling layer
        self.pool = torch.nn.MaxPool2d(2,2)
        
        
        # Fully connected layer
        self.fc1 = torch.nn.Linear(100*5*5, 800)   
        self.fc2 = torch.nn.Linear(800, 400)       
        self.fc3 = torch.nn.Linear(400, 136)
        
        self.drop1 = nn.Dropout(p=0.3)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
     
        # Flatten before passing to the fully-connected layers.
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop1(x)
        
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x