import matplotlib.pyplot as plt
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
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, item):
        image = self.features[item]
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)


train_images, temp_images, train_labels, temp_labels = train_test_split(images, labels, test_size=0.2, stratify=labels,
                                                                        random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5,
                                                                    stratify=temp_labels, random_state=42)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((75, 231)) #adjust in-features in first linear layer if removing Resize (longer computation time)
])

train_dataset = DataPreprocessor(train_images, train_labels, transform=data_transform)
val_dataset = DataPreprocessor(val_images, val_labels, transform=data_transform)
test_dataset = DataPreprocessor(test_images, test_labels, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = CNN().to(device)
print(model.train())

learning_rate = 0.001
epoch = 20
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for i in range(epoch):
    model.train()
    for image, target in train_loader:
        image = image.to(device)
        target = target.to(device)

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
        print(f'Validation Accuracy: {accuracy}')

# Testing (after training)
overall_predictions = []
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for image, target in test_loader:
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)
        overall_predictions.append(predicted.cpu().flatten().numpy())
    accuracy = total_correct / total_samples
    print(f'Testing Accuracy: {accuracy}')

final_predictions = []
for _ in overall_predictions:
    for x in _:
        final_predictions.append(x)

for _ in range(len(test_images)):
    plt.imshow(test_images[_])
    plt.xlabel(f"Correct author: {test_labels[_] + 1}, Author by model: {final_predictions[_] + 1}")
    plt.show()
    os.system('pause')
    plt.close()
