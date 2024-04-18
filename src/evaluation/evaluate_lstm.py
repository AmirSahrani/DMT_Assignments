import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import sys
import pdb
import matplotlib.pyplot as plt

sys.path.append('../models/') 
from lstm_data_loader import MoodDataset
from lstm_model import LSTMClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, test_loader, device):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    return true_labels, predictions

def load_test_data(csv_file, batch_size):
    test_dataset = MoodDataset(csv_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def plot_true_vs_predicted(true_labels, predictions, num_samples=100):
    # Reduce the number of samples for clarity in plotting
    true_labels = true_labels[:num_samples]
    predictions = predictions[:num_samples]
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_labels, label='True Mood', marker='o')
    plt.plot(predictions, label='Predicted Mood', marker='o')
    plt.title('True vs Predicted Mood')
    plt.xlabel('Samples')
    plt.ylabel('Mood')
    plt.legend()
    plt.show()

def main():
    test_csv_file = '../../data/preprocessed/test_final.csv' 
    batch_size = 32 
    input_size = 35
    hidden_size = 64 
    num_layers = 2 
    num_classes = 10

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load('../../data/models/lstm_classifier_full.pth', map_location=device))
    model.to(device)

    test_loader = load_test_data(test_csv_file, batch_size)

    true_labels, predictions = evaluate(model, test_loader, device)

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
