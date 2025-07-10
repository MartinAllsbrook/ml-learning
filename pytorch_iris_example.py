"""
Simple PyTorch Neural Network Example using Iris Dataset from scikit-learn
This example demonstrates:
1. Loading data from scikit-learn
2. Creating a simple neural network with PyTorch
3. Training the model
4. Evaluating the results
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the Iris dataset from scikit-learn
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))} ({iris.target_names})")


# Define a simple neural network
class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Initialize the model, loss function, and optimizer
model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\nModel Architecture:")
print(model)

# Training loop
num_epochs = 100
train_losses = []

print(f"\nStarting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    # Training accuracy
    train_outputs = model(X_train_tensor)
    _, train_predicted = torch.max(train_outputs.data, 1)
    train_accuracy = (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor)
    
    # Test accuracy
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"\nResults:")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Plot training loss
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot predictions vs actual for test set
plt.subplot(1, 2, 2)
test_predicted_np = test_predicted.numpy()
y_test_np = y_test_tensor.numpy()

scatter = plt.scatter(range(len(y_test_np)), y_test_np, c='blue', alpha=0.6, label='Actual')
plt.scatter(range(len(test_predicted_np)), test_predicted_np, c='red', alpha=0.6, label='Predicted')
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.yticks([0, 1, 2], iris.target_names)
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some sample predictions
print(f"\nSample predictions on test set:")
print(f"{'Actual':<12} {'Predicted':<12} {'Correct'}")
print("-" * 35)
for i in range(min(10, len(y_test))):
    actual = iris.target_names[y_test_np[i]]
    predicted = iris.target_names[test_predicted_np[i]]
    correct = "✓" if y_test_np[i] == test_predicted_np[i] else "✗"
    print(f"{actual:<12} {predicted:<12} {correct}")
