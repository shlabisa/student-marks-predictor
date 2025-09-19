import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import matplotlib.pyplot as plt
import random

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# ---------- 1. Load and Normalize Dataset ----------
class ExamDataset(Dataset):
    def __init__(self, dataframe, normalize=True):
        features = ['test1', 'test2', 'test3', 'assignment1', 'assignment2', 'project']
        target = 'exam'

        self.normalize = normalize
        self.feature_names = features

        X = dataframe[features].values.astype(np.float32)
        y = dataframe[target].values.astype(np.float32).reshape(-1, 1)

        self.X = torch.tensor(X, dtype=torch.float32) / 100
        """
        if normalize:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0) + 1e-8  # avoid division by 0
            self.X = torch.tensor((X - self.X_mean) / self.X_std, dtype=torch.float32)

            # Save normalization stats to JSON
            norm_stats = {
                'mean': self.X_mean.tolist(),
                'std': self.X_std.tolist()
            }
            os.makedirs('data', exist_ok=True)
            with open('data/normalization.json', 'w') as f:
                json.dump(norm_stats, f)
        else:
            self.X = torch.tensor(X, dtype=torch.float32)
        """

        self.y = torch.tensor(y / 100.0, dtype=torch.float32)  # normalize exam marks to [0,1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------- 2. Model ----------
class ExamPredictor(nn.Module):
    def __init__(self, input_size=6):
        super(ExamPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# ---------- 3. Train ----------
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                pred = model(X_val)
                val_loss += criterion(pred, y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# ---------- 4. Save / Load ----------
def save_model(model, path='exam_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def load_model(path='exam_model.pth', input_size=6):
    model = ExamPredictor(input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"📦 Model loaded from {path}")
    return model

# ---------- 5. Denormalize ----------
def denormalize(pred_tensor):
    """
    Converts a normalized prediction back to percentage (0-100 scale).
    """
    return pred_tensor.item() * 100

# ---------- 6. Predict ----------
def predict(input_tensor, model):
    """
    Predicts the exam mark for a given normalized input tensor using a model instance.
    Input shape: (1, 6)
    Output: Float in [0, 100]
    """
    with torch.no_grad():
        output = model(input_tensor)
    return denormalize(output)

# ---------- 7. Evaluate ----------
def evaluate_model(test_loader, model):
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"🧪 Test Set MSE Loss: {avg_loss:.4f}")

# ---------- 8. Visualize Prediction ----------
def visualize_prediction(X, model, Y_true=None):
    X = X.unsqueeze(0)  # shape (1, 6)

    Y_pred = predict(X, model)

    print("\n🔍 Test Sample Evaluation:")
    print(f"Input (normalized): {X.squeeze(0)}")
    print(f"Predicted Exam Mark: {Y_pred:.2f}")
    if Y_true is not None:
        Y_true = denormalize(Y_true)
        print(f"Actual Exam Mark   : {Y_true:.2f}")
        print(f"Loss (MSE)         : {((Y_pred - Y_true) ** 2):.4f}")

# ---------- 9. Plot ----------
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    plt.show()

# ---------- 10. Load Normalization Stat ----------
def load_normalization_stats(path='data/normalization.json'):
    with open(path, 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['mean'], dtype=np.float32)
    std = np.array(stats['std'], dtype=np.float32)
    return mean, std

# ---------- 11. Main ----------
def main():
    file_path = 'data/student_marks_500.csv'
    model_path = 'exam_model.pth'

    # Initialize the model
    model = ExamPredictor(input_size=6)
    """
    df = pd.read_csv(file_path)
    dataset = ExamDataset(df)
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = int(0.2 * total_size)
    val_size = total_size - train_size - test_size

    train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    # Train
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=256, lr=0.001)

    # Save and reuse
    save_model(model, model_path)

    # Plot loss
    plot_losses(train_losses, val_losses)

    # Evaluate
    evaluate_model(test_loader, model)
    
    # Visualize with random data
    idx = random.randint(0, len(test_data) - 1)
    X, Y = test_data[idx]
    visualize_prediction(X, model, Y_true=Y)
    """
    # Visualize with hard coded data
    input_data = torch.tensor([74,48,45,20,36,38]) / 100

    # Load model
    model = load_model(model_path)
    model.eval()

    # Predict and visualize
    visualize_prediction(input_data.squeeze(0), model)

if __name__ == '__main__':
    main()
