import os
import time
import torch
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torch import nn, optim
from sklearn.metrics import accuracy_score

# Define directories
spiral_dir = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Data_Spiral'
non_spiral_dir = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Data_Non_Spiral'

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset class
class SpiralDataset(Dataset):
    def __init__(self, spiral_dir, non_spiral_dir, transform=None):
        self.spiral_dir = spiral_dir
        self.non_spiral_dir = non_spiral_dir
        self.transform = transform
        self.spiral_images = [(os.path.join(spiral_dir, img), 0) for img in os.listdir(spiral_dir)]
        self.non_spiral_images = [(os.path.join(non_spiral_dir, img), 1) for img in os.listdir(non_spiral_dir)]
        self.images = self.spiral_images + self.non_spiral_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Create the dataset
dataset = SpiralDataset(spiral_dir, non_spiral_dir, transform=transform)

# Define train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16, pin_memory=True,prefetch_factor=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=16, pin_memory=True,prefetch_factor=8)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
# model = models.vgg16(pretrained=True)
# model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)  # Change output layer to match 2 classes
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.65,0.35]).to(device=device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

model_path = 'vgg_19'
# model_path = 'vgg_16'
# model_path = 'alexnet'


# Training the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    torch.save(model.state_dict(), f'{model_path}_{epoch+1}.pth')
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


# Evaluate the model
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
