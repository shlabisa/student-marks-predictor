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
import time

# ANSI escape codes for highlighting
GREEN = '\033[92m'
ENDC = '\033[0m'

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Define sequence names
SEQUENCE_MARKS = ['test1', 'test2', 'test3', 'assignment1', 'assignment2', 'project', 'exam']
ASSESSMENT_MARKS = SEQUENCE_MARKS[:-1] # test1...project (length 6)
EXAM_MARK = SEQUENCE_MARKS[-1]         # exam
MAX_SEQ_LEN = len(ASSESSMENT_MARKS)    # 6

# ---------- 1. Load and Normalize Dataset (Revised for Single Target with Augmentation) ----------
class MarkSequenceDataset(Dataset):
    
    augment = True # Class attribute set to True by default

    def __init__(self, dataframe):
        
        # 1. Extract and Normalize Original Data
        X_orig = dataframe[ASSESSMENT_MARKS].values.astype(np.float32)
        y_orig = dataframe[EXAM_MARK].values.astype(np.float32).reshape(-1, 1)

        # Normalize all marks by 100
        X_norm_orig = X_orig / 100.0
        y_norm_orig = y_orig / 100.0
        
        all_X = list(X_norm_orig)
        all_y = list(y_norm_orig)

        # 2. Data Augmentation
        if self.augment:
            print(f"📊 Generating augmented data (1 to 4 samples per original)...")
            num_original_samples = len(X_norm_orig)
            
            for i in range(num_original_samples):
                original_input = X_norm_orig[i]
                original_target = y_norm_orig[i]
                
                # Randomly decide to add 1 to 4 augmented data points
                num_augmentations = random.randint(1, 4)
                
                for _ in range(num_augmentations):
                    X_aug = self._augment(original_input.copy())
                    all_X.append(X_aug)
                    all_y.append(original_target) # Target remains the same
        
        # 3. Convert to Tensors
        self.X = torch.tensor(np.array(all_X), dtype=torch.float32).unsqueeze(-1) # Shape (N_total, 6, 1)
        self.y = torch.tensor(np.array(all_y), dtype=torch.float32)               # Shape (N_total, 1)
        
        # All sequence lengths are fixed at 6 (since we are only predicting the exam mark)
        self.L = torch.full((len(self.X),), MAX_SEQ_LEN, dtype=torch.long) # Shape (N_total,)
        
        print(f"Dataset Size: {len(X_norm_orig)} original samples. Total Size (incl. aug): {len(self.X)} samples.")


    def _augment(self, sequence):
        """
        Applies a random combination of augmentation techniques to the input sequence (normalized).
        
        Args:
            sequence (np.array): A 1D numpy array of the 6 assessment marks (normalized [0, 1]).
            
        Returns:
            np.array: The augmented 1D numpy array.
        """
        
        # Augmentation 1: Add small Gaussian noise (simulates measurement error/slight mark variation)
        if random.random() < 0.7: # 70% chance
            noise = np.random.normal(loc=0.0, scale=0.01, size=sequence.shape) # Scale is small (e.g., +/- 1 mark out of 100)
            sequence = sequence + noise
            
        # Augmentation 2: Time-Series Shift/Subsetting (simulates missing or swapped marks)
        if random.random() < 0.4: # 40% chance
            # Shift marks by one position and zero-fill the first/last mark
            if random.random() < 0.5:
                 # Shift right (losing the last mark, repeating the first/zeroing the first)
                 sequence = np.roll(sequence, 1)
                 sequence[0] = 0.0 # Fill start with 0
            else:
                 # Shift left (losing the first mark, zeroing the last)
                 sequence = np.roll(sequence, -1)
                 sequence[-1] = 0.0 # Fill end with 0

        # Augmentation 3: Multiply by a small random factor (simulates slight leniency/strictness)
        if random.random() < 0.5: # 50% chance
            # Factor between 0.95 and 1.05
            factor = np.random.uniform(0.95, 1.05)
            sequence = sequence * factor

        # Clamp values to the valid normalized range [0, 1]
        sequence = np.clip(sequence, 0.0, 1.0)
        
        return sequence

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return X: (6, 1), y: (1,), L: scalar (6)
        return self.X[idx], self.y[idx], self.L[idx]

# ---------- 2. Model (Unchanged) ----------
class MarkPredictorLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(MarkPredictorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, seq_lengths):
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        lstm_out_packed, (h_n, c_n) = self.lstm(x_packed)
        
        last_hidden_state = h_n[-1] # Output of the last layer
        
        output = self.fc(last_hidden_state)
        output = self.sigmoid(output)
        
        return output

# ---------- 3. Train (Unchanged) ----------
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch, L_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch, L_batch)
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
            for X_val, y_val, L_val in val_loader:
                pred = model(X_val, L_val)
                val_loss += criterion(pred, y_val).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

# ---------- 4. Save / Load (Unchanged) ----------
def save_model(model, path='lstm_mark_predictor_final.pth'):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def load_model(path='lstm_mark_predictor_final.pth'):
    model = MarkPredictorLSTM()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"📦 Model loaded from {path}")
    return model

# ---------- 5. Denormalize (Unchanged) ----------
def denormalize(pred_tensor):
    return pred_tensor.item() * 100

# ---------- 6. Predict (Updated for Fixed Sequence Input) ----------
def predict(input_sequence, model):
    """
    Predicts the next mark for a given sequence of 6 unnormalized marks.
    Input_sequence is a 1D tensor of *unnormalized* marks of length 6.
    """
    
    if len(input_sequence) != MAX_SEQ_LEN:
        raise ValueError(f"Input sequence must have exactly {MAX_SEQ_LEN} elements (assessments).")
        
    # 1. Normalize and structure the input
    normalized_seq = input_sequence / 100.0
    seq_len = MAX_SEQ_LEN
    
    # 2. Convert to tensor with required shape: (1, MAX_SEQ_LEN, 1)
    X = normalized_seq.unsqueeze(0).unsqueeze(-1)
    
    # 3. Sequence length tensor: (1,)
    L = torch.tensor([seq_len], dtype=torch.long)
    
    # 4. Prediction and time measurement for FPS
    start_time = time.time()
    with torch.no_grad():
        output = model(X, L)
    end_time = time.time()
    
    # 5. Denormalize and return with inference time
    return denormalize(output), (end_time - start_time)

# ---------- 7. Evaluate (Revised for Accuracy and FPS) ----------
def calculate_accuracy(preds_denorm, y_true_denorm, tolerance=5.0):
    """
    Calculates accuracy: percentage of predictions within a tolerance range of the true mark.
    """
    correct = torch.sum(torch.abs(preds_denorm - y_true_denorm) <= tolerance).item()
    total = len(y_true_denorm)
    accuracy = correct / total
    return accuracy

def evaluate_model(test_loader, model):
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    all_preds_denorm = []
    all_y_true_denorm = []
    total_inference_time = 0

    with torch.no_grad():
        for X_batch, y_batch, L_batch in test_loader:
            start_time = time.time()
            preds = model(X_batch, L_batch)
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
            
            all_preds_denorm.append(preds * 100)
            all_y_true_denorm.append(y_batch * 100)

    avg_loss = total_loss / total_samples
    
    all_preds_denorm = torch.cat(all_preds_denorm)
    all_y_true_denorm = torch.cat(all_y_true_denorm)
    
    accuracy = calculate_accuracy(all_preds_denorm, all_y_true_denorm, tolerance=5.0)
    
    if total_inference_time > 0:
        fps = total_samples / total_inference_time
    else:
        fps = float('inf')

    print(f"🧪 Test Set MSE Loss: {avg_loss:.4f}")
    print(f"🎯 Test Set Accuracy (±5 marks): {accuracy:.2f}")
    print(f"⚡ Inference Performance (FPS): {fps:.2f}")

# ---------- 8. Visualize Prediction (Enhanced with Highlight) ----------
def visualize_prediction(input_data_unnormalized, model, Y_true=None):
    
    try:
        Y_pred, inference_time = predict(input_data_unnormalized, model)
    except ValueError as e:
        print(f"\n🛑 Error for Visualization: {e}")
        return

    # Use ANSI code for green highlighting
    predicted_text = f"{GREEN}{Y_pred:.2f}{ENDC}"

    print("\n🔍 Test Sample Evaluation:")
    print(f"Input Assessments (unnormalized): {input_data_unnormalized.tolist()}")
    print(f"Predicted Exam Mark: {predicted_text}")
    print(f"Inference Time: {inference_time*1000:.3f} ms")

    if Y_true is not None:
        Y_true_denorm = denormalize(Y_true)
        print(f"Actual Exam Mark   : {Y_true_denorm:.2f}")
        print(f"Loss (MSE)         : {((Y_pred - Y_true_denorm) ** 2):.4f}")
        # Check accuracy for this single prediction
        is_accurate = '✅' if abs(Y_pred - Y_true_denorm) <= 5.0 else '❌'
        print(f"Accuracy (±5 marks): {is_accurate}")

# (9. Plot is unchanged)
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
    plt.savefig("loss_plot_lstm.png")
    plt.show()

# ---------- 11. Main (Revised Logic) ----------
def main():
    file_path = 'data/student_marks_500.csv'
    model_path = 'lstm_mark_predictor_final.pth' # Updated model path

    model = MarkPredictorLSTM()

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"🛑 Error: Data file not found at {file_path}. Please ensure it exists.")
        return

    # Use the simplified dataset for predicting the exam mark
    dataset = MarkSequenceDataset(df) 
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = int(0.2 * total_size)
    val_size = total_size - train_size - test_size

    if min(train_size, test_size, val_size) <= 0:
        print("🛑 Error: Not enough data to split into train, test, and validation sets.")
        return

    train_data, test_data, val_data = random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    # Train
    print("--- Starting Training (LSTM) ---")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=256, lr=0.001)

    # Save and reuse
    save_model(model, model_path)

    # Plot loss
    plot_losses(train_losses, val_losses)

    # Evaluate
    print("\n" + "="*50 + "\n")
    print("--- Model Evaluation on Test Set ---")
    evaluate_model(test_loader, model)
    
    print("\n" + "="*50 + "\n")
    print("--- Visualization: Random Samples from Training Set ---")

    # Visualize with 5 random samples from training set
    train_data_list = list(train_data)
    random_indices = random.sample(range(len(train_data_list)), k=5)
    
    for i, idx in enumerate(random_indices):
        X_train, Y_train, _ = train_data_list[idx]
        X_unnormalized_sequence = X_train.squeeze(-1) * 100 
        
        print(f"\n--- Random Sample {i+1} ---")
        visualize_prediction(X_unnormalized_sequence, model, Y_true=Y_train)
    
    print("\n" + "="*50 + "\n")
    print("--- Visualization: Hardcoded Samples with Different Trends ---")

    # Hard coded data (unnormalized, length must be 6)
    hardcoded_samples = [
        {
            "assessments": torch.tensor([85, 90, 88, 92, 89, 91], dtype=torch.float32), 
            "comment": "High and Consistent Marks (Strong PASS expected)"
        },
        {
            "assessments": torch.tensor([30, 45, 60, 75, 50, 65], dtype=torch.float32), 
            "comment": "Improving but Inconsistent Marks (Medium PASS expected)"
        },
        {
            "assessments": torch.tensor([95, 80, 65, 50, 35, 20], dtype=torch.float32), 
            "comment": "Steadily Declining Marks (Weak PASS/FAIL risk expected)"
        },
        # --- NEW EXAMPLES ---
        {
            "assessments": torch.tensor([20, 15, 25, 10, 30, 18], dtype=torch.float32),
            "comment": "Consistently Very Low Marks (Guaranteed FAIL expected)"
        },
        {
            "assessments": torch.tensor([30, 35, 40, 60, 75, 80], dtype=torch.float32),
            "comment": "Late Bloomer: Steadily and Dramatically Improving (High PASS expected)"
        },
        {
            "assessments": torch.tensor([85, 90, 80, 40, 30, 25], dtype=torch.float32),
            "comment": "Started Strong, Burned Out/Gave Up (FAIL risk expected)"
        },
        {
            "assessments": torch.tensor([50, 95, 30, 80, 45, 70], dtype=torch.float32),
            "comment": "Highly Volatile Marks (Unpredictable/Medium-High PASS expected)"
        }
    ]

    for i, sample in enumerate(hardcoded_samples):
        input_seq = sample["assessments"]
        print(f"\n--- Hardcoded Sample {i+1} ---")
        print(f"Trend: {sample['comment']}")
        visualize_prediction(input_seq, model, Y_true=None)

if __name__ == '__main__':
    main()