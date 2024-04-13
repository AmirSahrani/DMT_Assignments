import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from lstm_data_loader import MoodDataset
from lstm_classifier import LSTMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def cross_validate_and_save_models(csv_file, input_size, hidden_size, num_layers, num_classes, num_epochs=25, batch_size=32, learning_rate=0.001, save_path='../../data/models/'):
    dataset = MoodDataset(csv_file)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []  # evaluate on f1 score

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Starting fold {fold}')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop for the current fold
        for epoch in range(num_epochs):
            model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features.unsqueeze(1))  # Add a sequence dimension
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        fold_save_path = f"{save_path}lstm_fold_{fold}.pth"
        torch.save(model.state_dict(), fold_save_path)
        print(f'LSTM model for fold {fold} saved to {fold_save_path}')

        # Evaluate and calculate F1 score
        y_true = []
        y_pred = []
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='weighted')
        f1_scores.append(f1)
        print(f'Fold {fold} - F1 Score: {f1}')

    # Calculate average F1 score across folds
    avg_f1_score = np.mean(f1_scores)
    print(f'Average F1 Score: {avg_f1_score}')
    return f1_scores, avg_f1_score


def train_on_entire_dataset(csv_file, input_size, hidden_size, num_layers, num_classes, num_epochs, batch_size, learning_rate):
    dataset = MoodDataset(csv_file)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    final_model_path = '../../data/models/best_lstm_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Trained LSTM model saved at {final_model_path}')

    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
