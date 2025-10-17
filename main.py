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

# *** REVISED MAX_SEQ_LEN ***
MAX_SEQ_LEN = 8 # Set max sequence length to 8

# A helper function for padding, reusable by Dataset and Predict functions
def _pad_and_structure_input(sequence_unnormalized, max_len):
    """
    Normalizes, pads, and structures a variable-length sequence for the LSTM.
    
    Args:
        sequence_unnormalized (torch.Tensor or np.array): The mark sequence (unnormalized).
        max_len (int): The target padded length.
        
    Returns:
        tuple: (X_padded, L_actual) where X_padded is (1, max_len, 1) and L_actual is (1,).
    """
    
    # 1. Normalize
    normalized_seq = sequence_unnormalized / 100.0
    seq_len = len(normalized_seq)
    
    if seq_len > max_len:
        raise ValueError(f"Input sequence length ({seq_len}) exceeds MAX_SEQ_LEN ({max_len}).")

    # 2. Pad to max_len. Padding at the start (pre-padding) for sequence data is common.
    # Pad at the *end* (post-padding) here since we are predicting the final exam mark, 
    # and the last non-padded mark is the most important context.
    padding_needed = max_len - seq_len
    
    # Pad at the end (post-padding)
    padded_sequence = np.pad(normalized_seq.numpy(), (0, padding_needed), 
                             'constant', constant_values=(0.0, 0.0))
    
    # 3. Structure for LSTM: (1, max_len, 1)
    X = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    # 4. Sequence length tensor: (1,)
    L = torch.tensor([seq_len], dtype=torch.long)
    
    return X, L

# ---------- 1. Load and Normalize Dataset (Revised for Dynamic Input Size) ----------
class MarkSequenceDataset(Dataset):
    
    augment = True

    def __init__(self, dataframe):
        
        # 1. Extract Original Data (Assessments length 6)
        X_orig_unnorm = dataframe[ASSESSMENT_MARKS].values.astype(np.float32)
        y_orig = dataframe[EXAM_MARK].values.astype(np.float32).reshape(-1, 1)
        y_norm_orig = y_orig / 100.0
        
        all_X_unnorm = list(X_orig_unnorm)
        all_y = list(y_norm_orig)

        # 2. Data Augmentation (Augmented data remains length 6)
        if self.augment:
            print(f"📊 Generating augmented data (0 to 2 samples per original)...")
            num_original_samples = len(X_orig_unnorm)
            
            for i in range(num_original_samples):
                original_input = X_orig_unnorm[i]
                original_target = y_norm_orig[i]
                
                num_augmentations = random.randint(0, 2)
                
                for _ in range(num_augmentations):
                    # Augment unnormalized data
                    X_aug_unnorm = self._augment(original_input.copy()) 
                    all_X_unnorm.append(X_aug_unnorm)
                    all_y.append(original_target) 

        # 3. Process All Data for LSTM
        # We process each sample to pad it to MAX_SEQ_LEN (8)
        self.X_padded = []
        self.L_actual = []
        
        for X_unnorm in all_X_unnorm:
            X_tensor = torch.tensor(X_unnorm, dtype=torch.float32)
            # Use the new padding function. We ignore the (1,) batch dim here.
            X_padded_batch, L_batch = _pad_and_structure_input(X_tensor, MAX_SEQ_LEN)
            self.X_padded.append(X_padded_batch.squeeze(0)) # Shape (8, 1)
            self.L_actual.append(L_batch.squeeze(0))        # Shape (1,)

        self.X = torch.stack(self.X_padded, dim=0) # Shape (N_total, 8, 1)
        self.L = torch.stack(self.L_actual, dim=0) # Shape (N_total,)
        self.y = torch.tensor(np.array(all_y), dtype=torch.float32) # Shape (N_total, 1)
        
        print(f"Dataset Size: {len(X_orig_unnorm)} original samples. Total Size (incl. aug): {len(self.X)} samples.")
        print(f"Input tensor size is now: (N, {MAX_SEQ_LEN}, 1)")


    def _augment(self, sequence):
        """
        Applies augmentation to the unnormalized input sequence.
        """
        # Normalization is contained within augmentation logic for better control over noise scale
        sequence_norm = sequence / 100.0 
        
        # Augmentation 1: Add small Gaussian noise 
        if random.random() < 0.7: 
            noise = np.random.normal(loc=0.0, scale=0.01, size=sequence_norm.shape) 
            sequence_norm = sequence_norm + noise
            
        # Augmentation 2: Time-Series Shift/Subsetting 
        if random.random() < 0.4:
            if random.random() < 0.5:
                 sequence_norm = np.roll(sequence_norm, 1)
                 sequence_norm[0] = 0.0
            else:
                 sequence_norm = np.roll(sequence_norm, -1)
                 sequence_norm[-1] = 0.0

        # Augmentation 3: Multiply by a small random factor 
        if random.random() < 0.5: 
            factor = np.random.uniform(0.95, 1.05)
            sequence_norm = sequence_norm * factor

        # Clamp values to the valid normalized range [0, 1]
        sequence_norm = np.clip(sequence_norm, 0.0, 1.0)
        
        # Return unnormalized sequence
        return sequence_norm * 100.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return X: (8, 1), y: (1,), L: scalar (6)
        return self.X[idx], self.y[idx], self.L[idx]

# ---------- 2. Model (Updated input dimension) ----------
class MarkPredictorLSTM(nn.Module):
    # (Unchanged since it handles MAX_SEQ_LEN through pack_padded_sequence)
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
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
        
        last_hidden_state = h_n[-1]
        
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

# ---------- 6. Predict (Updated for Dynamic Sequence Input) ----------
def predict(input_sequence, model):
    """
    Predicts the next mark for a given sequence of 1 to MAX_SEQ_LEN unnormalized marks.
    """
    
    seq_len = len(input_sequence)
    if seq_len < 1 or seq_len > MAX_SEQ_LEN:
        raise ValueError(f"Input sequence length ({seq_len}) must be between 1 and {MAX_SEQ_LEN}.")
        
    # Use the helper function to pad and structure the input
    X, L = _pad_and_structure_input(input_sequence, MAX_SEQ_LEN)
    
    # 4. Prediction and time measurement for FPS
    start_time = time.time()
    with torch.no_grad():
        output = model(X, L)
    end_time = time.time()
    
    # 5. Denormalize and return with inference time
    return denormalize(output), (end_time - start_time)

# ---------- 7. Evaluate (Unchanged) ----------
def calculate_accuracy(preds_denorm, y_true_denorm, tolerance=5.0):
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

# ---------- 8. Visualize Prediction (Unchanged) ----------
def visualize_prediction(input_data_unnormalized, model, Y_true=None):
    
    try:
        Y_pred, inference_time = predict(input_data_unnormalized, model)
    except ValueError as e:
        print(f"\n🛑 Error for Visualization: {e}")
        return

    predicted_text = f"{GREEN}{Y_pred:.2f}{ENDC}"

    print("\n🔍 Test Sample Evaluation:")
    print(f"Input Assessments (unnormalized) [Len: {len(input_data_unnormalized)}]: {input_data_unnormalized.tolist()}")
    print(f"Predicted Exam Mark: {predicted_text}")
    print(f"Inference Time: {inference_time*1000:.3f} ms")

    if Y_true is not None:
        Y_true_denorm = denormalize(Y_true)
        print(f"Actual Exam Mark   : {Y_true_denorm:.2f}")
        print(f"Loss (MSE)         : {((Y_pred - Y_true_denorm) ** 2):.4f}")
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

# ---------- 11. Main (Revised Logic for Dynamic Input) ----------
def main():
    file_path = 'data/student_marks_500.csv'
    model_path = 'lstm_mark_predictor_final.pth'

    model = MarkPredictorLSTM()

    if os.path.exists(model_path):
        model = load_model(model_path)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"🛑 Error: Data file not found at {file_path}. Please ensure it exists.")
        return

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
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=128, lr=0.001) # Reduced epochs for faster output

    # Save and reuse
    save_model(model, model_path)

    # Plot loss
    plot_losses(train_losses, val_losses)

    # Evaluate
    print("\n" + "="*50 + "\n")
    print("--- Model Evaluation on Test Set ---")
    evaluate_model(test_loader, model)
    
    print("\n" + "="*50 + "\n")
    print("--- Visualization: Dynamic Hardcoded Samples (Len 2 to 8) ---")

    # Hard coded data with DYNAMIC lengths (max length is 8, but actual input is variable)
    hardcoded_dynamic_samples = [
        {
            "assessments": torch.tensor([85, 90], dtype=torch.float32), 
            "comment": "Length 2: Strong Start (Early prediction)"
        },
        {
            "assessments": torch.tensor([30, 45, 60, 75], dtype=torch.float32), 
            "comment": "Length 4: Improving trend (Mid-point prediction)"
        },
        {
            "assessments": torch.tensor([95, 80, 65, 50, 35, 20], dtype=torch.float32), 
            "comment": "Length 6: Declining trend (Full assessment prediction)"
        },
        {
            "assessments": torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.float32),
            "comment": f"Length 8: Hypothetical Full Max-Length Sequence (Maximum Context)"
        }
    ]

    for i, sample in enumerate(hardcoded_dynamic_samples):
        input_seq = sample["assessments"]
        print(f"\n--- Hardcoded Sample {i+1} ---")
        print(f"Trend: {sample['comment']}")
        # The prediction function now handles the variable length
        visualize_prediction(input_seq, model, Y_true=None)

if __name__ == '__main__':
    main()