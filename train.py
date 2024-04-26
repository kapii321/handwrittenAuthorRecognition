import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from torch import optim
import torch.nn as nn
from model import CNN
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

images = []
labels = []
dataPath = r'C:\Users\Kacper\PycharmProjects\aiProjectNew\AllWords'
subFolder = os.listdir(dataPath)
for folder in subFolder:
    label = subFolder.index(folder)
    path = os.path.join(dataPath, folder)
    for imglist in os.listdir(path):
        image = cv2.imread(os.path.join(path, imglist))
        images.append(image)
        labels.append(label)


class DataPreprocessor(Dataset):
    def __init__(self, features, labels, num_classes, transform=None):
        self.features = features
        self.labels = labels
        self.num_classes = num_classes
        self.transform = transform

    def __getitem__(self, item):
        image = self.features[item]
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)

        target = torch.zeros(self.num_classes)
        target[label] = 1

        return image, target

    def __len__(self):
        return len(self.labels)


train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, stratify=labels,
                                                                      random_state=42)

data_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DataPreprocessor(train_images, train_labels, num_classes=len(subFolder), transform=data_transform)
val_dataset = DataPreprocessor(val_images, val_labels, num_classes=len(subFolder), transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = CNN(num_authors=8).to(device)
print(model.train())

learning_rate = 0.001
epoch = 100
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for i in range(epoch):
    model.train()
    for image, target in train_loader:
        image = image.to(device)
        target = target.to(device)

        target = target.long()

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epochs: {i + 1} out of {epoch} || Training Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for image, target in val_loader:
            image = image.to(device)
            target = target.to(device)

            output = model(image)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
        accuracy = total_correct / total_samples
        print(f'Accuracy: {accuracy}')
