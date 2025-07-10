"""
Simple PyTorch Neural Network Example - Non-Interactive Version
This is a simplified version that saves plots instead of displaying them interactively.
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

# Use non-interactive backend for matplotlib
plt.switch_backend('Agg')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the Iris dataset
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create data loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))} ({iris.target_names})")


# Simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        
    def forward(self, x):
        return self.network(x)


# Initialize model, loss, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\nModel Architecture:")
print(model)

# Training loop
num_epochs = 100
train_losses = []

print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 25 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    _, train_predicted = torch.max(train_outputs.data, 1)
    train_accuracy = (train_predicted == y_train_tensor).sum().item() / len(y_train_tensor)
    
    test_outputs = model(X_test_tensor)
    _, test_predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"\n{'='*50}")
print(f"FINAL RESULTS:")
print(f"{'='*50}")
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Save training loss plot
plt.figure(figsize=(8, 6))
plt.plot(train_losses, 'b-', linewidth=2)
plt.title('Training Loss Over Time', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
print(f"\nTraining loss plot saved as 'training_loss.png'")

# Print detailed predictions
print(f"\n{'='*50}")
print(f"SAMPLE PREDICTIONS:")
print(f"{'='*50}")
test_predicted_np = test_predicted.numpy()
y_test_np = y_test_tensor.numpy()

correct_predictions = 0
print(f"{'Index':<6} {'Actual':<12} {'Predicted':<12} {'Status'}")
print("-" * 45)

for i in range(len(y_test)):
    actual = iris.target_names[y_test_np[i]]
    predicted = iris.target_names[test_predicted_np[i]]
    is_correct = y_test_np[i] == test_predicted_np[i]
    status = "✓ Correct" if is_correct else "✗ Wrong"
    
    if is_correct:
        correct_predictions += 1
    
    print(f"{i:<6} {actual:<12} {predicted:<12} {status}")

print(f"\nCorrect predictions: {correct_predictions}/{len(y_test)}")
print(f"Accuracy: {correct_predictions/len(y_test)*100:.1f}%")

# Print model summary
print(f"\n{'='*50}")
print(f"MODEL SUMMARY:")
print(f"{'='*50}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Model size: ~{total_params * 4 / 1024:.2f} KB")  # Assuming float32

print(f"\n{'='*50}")
print("PyTorch + scikit-learn example completed successfully!")
print(f"{'='*50}")
