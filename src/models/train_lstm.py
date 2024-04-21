import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from lstm_model import LSTMClassifier, LSTMRegressor
from lstm_data_loader import MoodDataset
import pdb


def calculate_class_weights(dataset, beta=0.83):
    """
    Calculate class weights using the "Effective Number of Samples" method.
    Reference: https://arxiv.org/abs/1901.05555

    Args:
    - dataset: the dataset to calculate weights for.
    - beta: hyperparameter for class weights calculation, controlling the strength of the weighting.

    Returns:
    - torch.tensor: class weights.
    """
    labels = np.array([data[1] for data in dataset])
    num_classes = 10  # Adjust if different
    class_counts = np.bincount(labels, minlength=num_classes)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Effective number of samples calculation
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)

    # Normalizing the weights to make the sum equal to the number of classes
    weights = (weights / weights.sum()) * num_classes

    # Convert to torch tensor
    class_weights = torch.tensor(weights, dtype=torch.float32)

    return class_weights


def calculate_regression_weights(y_train, num_bins=10, smoothing_factor=0.1, majority_penalty=2):
    bin_edges = np.histogram_bin_edges(y_train, bins=num_bins)
    bin_counts = np.histogram(y_train, bins=bin_edges)[0]

    weights = 1.0 / (np.power(bin_counts, majority_penalty) + smoothing_factor)

    # To prevent too much penalty on the majority class
    min_weight = np.percentile(weights, 50)
    weights = np.maximum(weights, min_weight)

    # Normalize the weights to maintain the loss scale
    weights /= np.mean(weights)

    return torch.tensor(weights, dtype=torch.float32), bin_edges


class WeightedMAELoss(nn.Module):
    def __init__(self, bin_edges, bin_weights):
        super().__init__()
        self.bin_edges = bin_edges
        self.bin_weights = bin_weights

    def forward(self, predicted, target):
        bin_indices = np.digitize(target.cpu().numpy(), self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_weights) - 1)
        weights = self.bin_weights[bin_indices].to(predicted.device)
        loss = torch.abs(predicted.squeeze() - target) * weights
        return loss.mean()


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

    class_weights = calculate_class_weights(dataset) if model_type == 'classification' else None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

        if model_type == 'classification':
            model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        elif model_type == 'regression':
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
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class_weights = calculate_class_weights(dataset) if model_type == 'classification' else None

    if model_type == 'classification':
        model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif model_type == 'regression':
        model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device)
        y_train = np.array([data[1] for data in dataset])
        weights, bin_edges = calculate_regression_weights(y_train)
        criterion = WeightedMAELoss(bin_edges, weights).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Full Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), f"../../data/models/lstm_{model_type}_optimized.pth")


def main():
    n_splits = 5
    batch_size = 8
    num_epochs = 63
    learning_rate = 0.01959
    hidden_size = 32
    num_layers = 2
    num_classes = 10  # For classifier
    output_size = 1   # For regressor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = 'classification'  # or 'regression'
    csv_file = '../../data/preprocessed/train_classification.csv' if model_type == 'classification' else '../../data/preprocessed/train_regression.csv'
    train_dataset = MoodDataset(csv_file=csv_file, mode=model_type)
    input_size = train_dataset.get_num_features()
    print(f"Number of features: {input_size}")

    # Cross-validate using Time Series Split
    # fold_results = cross_validate(train_dataset, model_type, n_splits, batch_size, input_size, hidden_size, num_layers, num_classes if model_type == 'classification' else output_size, num_epochs, learning_rate, device)
    # average_validation_loss = np.mean(fold_results)
    # print(f"Cross-validation results: {fold_results}")
    # print(f"Average validation loss: {average_validation_loss}")

    # Full training on the entire dataset
    full_training(train_dataset, model_type, batch_size, input_size, hidden_size, num_layers, num_classes if model_type == 'classification' else output_size, num_epochs, learning_rate, device)


if __name__ == '__main__':
    main()
