import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import wandb
print(torch.cuda.device_count())
wandb.init(project="mnist-pytorch-demo", name="cnn-logging_2")

learning_rate = 1e-3
num_epochs = 10
batch_size = 64
val_ratio = 0.1
seed = 42
torch.manual_seed(seed)


data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

n_total = len(data)
n_val = int(n_total * val_ratio)
n_train = n_total - n_val
train_dataset, val_dataset = random_split(data, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool2(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        count += target.size(0)
    avg_loss = total_loss / len(train_loader)
    acc = correct / count
    print(f"[Train] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    wandb.log({"train/loss": avg_loss, "train/acc": acc, "epoch": epoch})

@torch.no_grad()
def validate(model, device, val_loader, epoch):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        count += target.size(0)
    avg_loss = total_loss / len(val_loader)
    acc = correct / count
    print(f"[Valid] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    wandb.log({"valid/loss": avg_loss, "valid/acc": acc, "epoch": epoch})

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    wandb.log({"test/loss": test_loss, "test/acc": correct / len(test_loader.dataset)})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    validate(model, device, val_loader, epoch)

test(model, device, test_loader)

torch.save(model.state_dict(), 'mnist_cnn.pth')
model2 = ConvNet().to(device)
model2.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model2.eval()
test(model2, device, test_loader)
