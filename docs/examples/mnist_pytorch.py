import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def load_mnist(batch_size=64, data_dir='./data'):
    # Define transformations to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the MNIST training and testing datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    # Create data loaders for training and testing datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Define a simple feedforward neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    return test_loss / len(test_loader), accuracy

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define batch size, learning rate, and number of epochs
    batch_size = 64
    learning_rate = 0.001
    epochs = 10
    
    # Check if GPU is available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST data
    train_loader, test_loader = load_mnist(batch_size=batch_size, data_dir='./data')
    
    # Create a neural network model
    model = Net().to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
