import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
import optuna
from lstm_model import LSTMClassifier, LSTMRegressor
from lstm_data_loader import MoodDataset 

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    total_val_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_val_loss / total_samples

def objective(trial, device, dataset, model_type):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 50, 300)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_losses = []

    for train_idx, val_idx in tscv.split(dataset):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        input_size = dataset.get_num_features()
        output_size = 1 if model_type == 'regression' else 10

        if model_type == 'classification':
            model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
            criterion = nn.CrossEntropyLoss()
        elif model_type == 'regression':
            model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device)
            criterion = nn.L1Loss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate on the current fold
        val_loss = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)
        fold_losses.append(val_loss)

    # Average loss across folds
    avg_loss = np.mean(fold_losses)
    trial.report(avg_loss, len(fold_losses))

    # Prune unpromising trials
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return avg_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type = 'regression'
    dataset = MoodDataset(csv_file=f"../../data/preprocessed/train_{model_type}.csv", mode=model_type)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, device, dataset, model_type), n_trials=100)

    # Save study results to a CSV file
    df = study.trials_dataframe()
    df.to_csv(f"../../data/hyperparam_{model_type}_result.csv", index=False)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial
    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")

if __name__ == '__main__':
    main()