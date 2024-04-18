import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from lstm_model import LSTMClassifier, LSTMRegressor
from lstm_data_loader import MoodDataset
import pdb

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() 
    return total_loss / len(val_loader)

def cross_validate(dataset, model_type, n_splits, batch_size, input_size, hidden_size, num_layers, output_size, num_epochs, learning_rate, device):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        if model_type == 'classifier':
            model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
            criterion = nn.CrossEntropyLoss()
        elif model_type == 'regressor':
            model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device)
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss = validate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        fold_results.append(val_loss)
        torch.save(model.state_dict(), f"../../data/models/lstm_{model_type}_fold_{fold + 1}.pth")

    return fold_results

def full_training(dataset, model_type, batch_size, input_size, hidden_size, num_layers, output_size, num_epochs, learning_rate, device):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if model_type == 'classifier':
        model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
    elif model_type == 'regressor':
        model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Full Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), f"../../data/models/lstm_{model_type}_full.pth")
    print(f"Full model saved as lstm_{model_type}_full.pth")

def main():
    n_splits = 5
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    hidden_size = 64
    num_layers = 2
    num_classes = 10  # For classifier
    output_size = 1   # For regressor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MoodDataset(csv_file='../../data/preprocessed/train_final.csv')
    input_size = train_dataset.get_num_features()
    print(f"Number of features: {input_size}")

    model_type = 'classifier'  # or 'regressor'

    # Cross-validate using Time Series Split
    fold_results = cross_validate(train_dataset, model_type, n_splits, batch_size, input_size, hidden_size, num_layers, num_classes if model_type == 'classifier' else output_size, num_epochs, learning_rate, device)
    average_validation_loss = np.mean(fold_results)
    print(f"Cross-validation results: {fold_results}")
    print(f"Average validation loss: {average_validation_loss}")

    # Full training on the entire dataset
    full_training(train_dataset, model_type, batch_size, input_size, hidden_size, num_layers, num_classes if model_type == 'classifier' else output_size, num_epochs, learning_rate, device)

if __name__ == '__main__':
    main()