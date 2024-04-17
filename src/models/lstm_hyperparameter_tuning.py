import optuna
import csv
import numpy as np
from lstm_model import MoodDataset, LSTMClassifier
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch
from sklearn.model_selection import KFold
from optuna.trial import TrialState

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lstm_hyperparameter_tuning(csv_file, features, num_trials=100, num_folds=5, csv_out='hyperparam_results.csv'):
    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 16, 128)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        sequence_length = trial.suggest_int('sequence_length', 3, 10)
        num_epochs = trial.suggest_int('num_epochs', 5, 20)
        
        dataset = MoodDataset(csv_file, features, sequence_length)
        input_size = dataset.features.shape[-1]
        num_classes = 10 
        
        fold_losses = []
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

            # Model setup
            model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train the model
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for data, labels in train_loader:
                    data, labels = data.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)

            # Evaluate the model on the validation fold
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

            fold_losses.append(validation_loss / len(test_loader))
            trial.report(np.mean(fold_losses), fold)

            # stop trial if unlikely to lead to a better result than the best one obtained
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        # The objective is to minimize the average loss across the folds
        return np.mean(fold_losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    # Save results to CSV
    with open(csv_out, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_number', 'hidden_size', 'num_layers',
                         'batch_size', 'learning_rate', 'sequence_length', 'num_epochs', 'avg_fold_loss'])
        for trial in study.trials:
            if trial.state == TrialState.COMPLETE:
                writer.writerow([trial.number, trial.params['hidden_size'],
                                 trial.params['num_layers'], trial.params['batch_size'],
                                 trial.params['learning_rate'], trial.params['sequence_length'],
                                 trial.params['num_epochs'], trial.value])

    return study

if __name__ == '__main__':
    features = [
                'activity', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.social', 'appCat.travel',
                'appCat.utilities', 'appCat.weather', 'call', 'circumplex.arousal', 'circumplex.valence',
                'screen', 'sms', 'hour', 'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos'
            ]
    
    lstm_hyperparameter_tuning(
        csv_file='../../data/preprocessed/train_set.csv',
        features=features,
        num_trials=100,
        csv_out='../../data/hyperparam_result.csv'
    )