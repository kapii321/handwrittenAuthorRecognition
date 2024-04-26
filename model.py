import torch
import torch.nn as nn
import torch.nn.functional as F


def activation_layer(activation: str='relu', alpha: float=0.1, inplace: bool = True):
    if activation == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, skip_conv=True, dropout=0.2,
                 activation='leaky_relu', padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = activation_layer(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                               padding=padding)  # Stride is always 1
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout)

        # Adjusting the shortcut connection to match the dimensions of the main path
        if skip_conv:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),  # Adjust spatial dimensions
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()  # Identity mapping if dimensions match
        else:
            self.shortcut = nn.Identity()  # No skip connection

        self.activation2 = activation_layer(activation)

    def forward(self, x):
        jump = self.shortcut(x)  # Apply shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += jump  # Perform element-wise addition with shortcut connection

        out = self.activation2(out)

        out = self.dropout(out)

        return out


class CNN(nn.Module):

    def __init__(self,num_authors: int, activation: str='leaky_relu', dropout: float=0.2):
        super(CNN, self).__init__()

        self.rb1 = ResidualBlock(3,16, skip_conv=True, stride=1, activation=activation, dropout=dropout)
        self.rb2 = ResidualBlock(16,32, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb3 = ResidualBlock(32,64, skip_conv=True, stride=2, activation=activation, dropout=dropout)
        self.rb4 = ResidualBlock(64,64, skip_conv=True, stride=1, activation=activation,dropout=dropout)

        self.lstm = nn.LSTM(64,128, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout)

        self.output = nn.Linear(256, num_authors)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        #normalization if needed
        '''
        images_float = images / 255.0
        images_float = images_float.permute(0,3,1,2)
        '''

        #change images to images_float if using normalization here
        x = self.rb1(images)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.output(x)

        # was log_softmax but not needed for crossEntropyLoss, it is used only for
        # CTCloss and we dont need it for author recognition, see if regular softmax works okay
        # also test without softmax activation here, CrossEntropy uses it internally automatically


        #dim =2 because of reshaping for lstm, remove if doesnt work properly and dont use softmax here
        x = F.softmax(x, dim=2)

        return x
