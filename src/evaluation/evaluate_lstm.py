from lstm_model import LSTMClassifier, LSTMRegressor
from lstm_data_loader import MoodDataset
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_squared_error, mean_absolute_error
import sys
import matplotlib.pyplot as plt
import numpy as np

if isinstance(os.path, str):
    pass
else:
    os._path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    print('File is lstm_eval, loc is:', os._path)


# add ../models to os.path
sys.path.append('../data/models/')
sys.path.append('../models/')


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
    bars1 = ax.bar(x - width / 2, counts.values(), width, label='True Mood', color='blue')
    bars2 = ax.bar(x + width / 2, counts_pred.values(), width, label='Predicted Mood', color='orange')

    ax.set_title('True vs Predicted Mood')
    ax.set_xlabel('Mood Categories')
    ax.set_ylabel('Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(counts.keys())
    ax.legend()

    # This will display the plot in a manner similar to the image you uploaded
    fig.tight_layout()
    plt.show()


def plot_true_vs_predicted_histogram(y_test, y_pred):
    # Define bin edges to match the categories in the histogram
    bins = np.arange(-0.5, 11, 1)  # Bin edges from -0.5 to 10.5

    # Digitize categorizes the data into bins
    y_test_bins = np.digitize(y_test, bins) - 1  # -1 to convert bins to 0-based index
    y_pred_bins = np.digitize(y_pred, bins) - 1

    # Count occurrences in each bin
    counts = np.zeros(len(bins) - 1)
    counts_pred = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        counts[i] = np.sum(y_test_bins == i)
        counts_pred[i] = np.sum(y_pred_bins == i)

    # Set up the plot
    x = np.arange(len(bins) - 1)  # The label locations
    width = 0.4  # Width of the bars

    plt.figure(figsize=(10, 8))
    plt.bar(x - width / 2, counts, width, label='True Mood', color='blue')
    plt.bar(x + width / 2, counts_pred, width, label='Predicted Mood', color='orange')
    plt.title('True vs Predicted Mood with LSTM Regression')
    plt.xlabel('Mood Score')
    plt.ylabel('Frequency')
    plt.xticks(x, labels=[str(i) for i in range(11)])
    plt.legend()
    plt.show()


def main(model_type='regression', optimized=False, visualize=True):

    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    test_csv_file = f'../data/preprocessed/test_{model_type}.csv'
    abs_path = os.path.join(current_path, test_csv_file)
    test_csv_file = abs_path
    batch_size = 32

    if model_type == 'classification':
        input_size = 38
        hidden_size = 64
        num_layers = 2 + optimized
        num_classes = 10
        model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    elif model_type == 'regression':
        input_size = 38
        hidden_size = 64
        num_layers = 2 + optimized
        output_size = 1
        model = LSTMRegressor(input_size, hidden_size, num_layers, output_size)

    optimize = '_optimized' if optimized else '_unoptimized'
    model.load_state_dict(torch.load(f'../../data/models/lstm_{model_type}{optimize}.pth', map_location=device))
    model.to(device)

    test_loader = load_test_data(test_csv_file, batch_size, mode=model_type)

    true_labels, predictions = evaluate(model, test_loader, device, model_type)
    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    print(f'MSE: {mse:.3f}')
    print(f'MAE: {mae:.3f}')

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
        if visualize:
            plot_true_vs_predicted(true_labels, predictions)
    else:
        if visualize:
            plot_true_vs_predicted_histogram(true_labels, predictions)

    return true_labels, predictions


if __name__ == '__main__':
    main('classification', True, True)
