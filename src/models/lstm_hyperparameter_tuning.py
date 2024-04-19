import torch
import torch.nn as nn
import numpy as np
import optuna
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from lstm_model import LSTMClassifier, LSTMRegressor 
from lstm_data_loader import MoodDataset

def objective(trial, device, dataset, model_type, n_splits):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    num_epochs = trial.suggest_int('num_epochs', 5, 50)
    input_size = dataset.get_num_features()

    output_size = 10 if model_type == 'classification' else 1 

    model = (LSTMClassifier(input_size, hidden_size, num_layers, output_size, dropout=dropout) if model_type == 'classification' 
             else LSTMRegressor(input_size, hidden_size, num_layers, output_size, dropout=dropout))
    model.to(device)

    criterion = nn.CrossEntropyLoss() if model_type == 'classification' else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Perform a cross-validation on the Time Series Split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_losses = []
    
    for train_idx, val_idx in tscv.split(dataset):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        for _ in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        val_losses.append(total_val_loss / len(val_loader))

    average_val_loss = np.mean(val_losses)
    trial.report(average_val_loss, n_splits)
    return average_val_loss

def run_optuna(model_type, device, dataset):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, device, dataset, model_type, n_splits=5), n_trials=50)

    # Save study results to a CSV
    df = study.trials_dataframe()
    df.to_csv(f'optuna_results_{model_type}.csv', index=False)

    print(f"Best Params: {study.best_params}")
    print(f"Best Validation Loss: {study.best_value}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = MoodDataset(csv_file='../../data/preprocessed/train_final.csv', mode='classification') 

run_optuna('classification', device, train_dataset)

