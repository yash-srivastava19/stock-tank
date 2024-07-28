import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout_prob)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out

def load_data(symbol):
    """
    Load the processed training data.
    """
    return pd.read_csv(f"data/processed/{symbol}_train_data.csv", index_col=0, parse_dates=True)

def prepare_sequences(data, seq_length):
    """
    Prepare sequences for LSTM model.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:(i + seq_length)].values)
        y.append(data.iloc[i + seq_length]['Target'])
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, device):
    """
    Train the LSTM model.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # save_model_version(model, f"models/{args.symbol}_lstm_model.pth")
            # torch.save(model.state_dict(), f"models/{args.symbol}_lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

def main(args):
    data = load_data(args.symbol)
    X, y = prepare_sequences(data, args.seq_length)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    input_dim = X_train.shape[2]
    hidden_dim = args.hidden_dim
    output_dim = 1
    dropout_prob = args.dropout_prob
    
    model = LSTMModel(input_dim, hidden_dim, output_dim, dropout_prob)
    
    train_model(model, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model for {args.symbol} has been trained and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for time series prediction.")
    parser.add_argument('symbol', type=str, help="The stock symbol or dataset identifier.")
    parser.add_argument('seq_length', type=int, help="The sequence length for LSTM input.")
    parser.add_argument('--hidden_dim', type=int, default=50, help="Number of hidden units in LSTM layers.")
    parser.add_argument('--dropout_prob', type=float, default=0.2, help="Dropout probability.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    
    args = parser.parse_args()
    main(args)