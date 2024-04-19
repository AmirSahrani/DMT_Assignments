import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, mean_absolute_error
import pandas as pd
import sys
import pdb
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../models/')
from lstm_data_loader import MoodDataset
from lstm_model import LSTMClassifier, LSTMRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, test_loader, device, model_type):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if model_type == 'classification':
                _, predicted = torch.max(outputs, 1)
            elif model_type == 'regression':
                predicted = outputs.squeeze()

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    return true_labels, predictions

def load_test_data(csv_file, batch_size, mode='classification'):
    test_dataset = MoodDataset(csv_file, mode=mode)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def plot_true_vs_predicted(y_test, y_pred):
    counts = {i: 0 for i in range(11)}
    counts_pred = {i: 0 for i in range(11)}
    for i in range(len(y_test)):
        counts[y_test[i]] += 1
        counts_pred[y_pred[i]] += 1

    x = np.arange(0, 11)
    width = 0.4  # width of each bar

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, counts.values(), width, label='True Mood', color='blue')
    bars2 = ax.bar(x + width/2, counts_pred.values(), width, label='Predicted Mood', color='orange')

    ax.set_title('True vs Predicted Mood')
    ax.set_xlabel('Mood Categories')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(counts.keys())
    ax.legend()

    # This will display the plot in a manner similar to the image you uploaded
    fig.tight_layout()
    plt.show()

def main():
    test_csv_file = '../../data/preprocessed/test_classification.csv'
    batch_size = 32
    model_type = 'classification'

    if model_type == 'classification':
        input_size = 38
        hidden_size = 64
        num_layers = 3
        num_classes = 10
        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    elif model_type == 'regression':
        input_size = 36
        hidden_size = 64
        num_layers = 2
        output_size = 1
        model = LSTMRegressor(input_size, hidden_size, num_layers, output_size)

    model.load_state_dict(torch.load(f'../../data/models/lstm_{model_type}_optimized_weighted.pth', map_location=device))
    model.to(device)

    test_loader = load_test_data(test_csv_file, batch_size, mode=model_type)

    true_labels, predictions = evaluate(model, test_loader, device, model_type)
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    print(f'MSE: {mse:.2f}')
    print(f'MAE: {mae:.2f}')

    if model_type == 'classification':
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(classification_report(true_labels, predictions))

    plot_true_vs_predicted(true_labels, predictions)

if __name__ == '__main__':
    main()
