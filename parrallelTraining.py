import torch
from tqdm import tqdm
from torch import nn
from models import HeavyModel
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from yaml import safe_load

try:
    with open("config.yaml", "r") as file:
        config = safe_load(file)
except FileNotFoundError:
    raise FileNotFoundError("The config.yaml file was not found. Check the path.")


# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeavyModel().to(device)  # TODO: Try .half*()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CIFAR-10 data loading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
train_dataset = Subset(train_dataset, range(16384))
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)


def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )


if __name__ == "__main__":
    train_model(model, train_loader, criterion, optimizer, epochs=1)
