import torch.nn as nn



class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=190464, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=8)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool(x)
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x