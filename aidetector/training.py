"""
This module contains the functions related to training and saving models based on input data.

Imports:
    pandas: Library for data manipulation and analysis.
    torch: PyTorch library for tensor computations and neural networks.
    sklearn.model_selection: Scikit-learn library for model selection and evaluation.
    torch.optim: PyTorch library for optimization algorithms.
    pickle: Python's object serialization module.
"""

import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import pickle


def detect_delimiter(filename, col1, col2):
    with open(filename, 'r') as f:
        first_line = f.readline()
    parts = first_line.split(col1)
    if len(parts) < 2:
        raise ValueError(f"Couldn't find column name {col1} in the first line of the file")
    second_part = parts[1]
    parts = second_part.split(col2)
    if len(parts) < 2:
        raise ValueError(f"Couldn't find column name {col2} in the first line of the file")
    delimiter = parts[0]
    return delimiter


#Data loading...
def load_data(inputfile, percentsplit=0.2,classificationlabel='classification',textlabel='text'):
    """
    Load data from a CSV file and split it into training and testing sets.

    Parameters:
    inputfile (str): Path to the CSV file.
    percentsplit (float, optional): Fraction of the data to be used as a test set. Default is 0.2.
    classificationlabel (str, optional): Column label for classification. Default is 'classification'.
    textlabel (str, optional): Column label for text. Default is 'text'.

    Returns:
    tuple: Four pandas Series objects for training text, testing text, training labels, and testing labels.
    """
    delimeter=detect_delimiter(inputfile,classificationlabel,textlabel)
    df = pd.read_csv(
        inputfile,
        sep=delimeter
    )
    traintxt, testtxt, trainlbl, testlbl = train_test_split(
        df[textlabel],
        df[classificationlabel],
        test_size=percentsplit
    )

    return traintxt, testtxt, trainlbl, testlbl


#train the model...
def train_model(model_training, sequences_training, labels_training, sequences_testing, labels_testing, epochs=100, lowerbound=0.4, upperbound=0.6):
    """
    Train a model on training sequences and labels, with validation on testing sequences and labels.

    Parameters:
    model_training (nn.Module): The model to be trained.
    sequences_training (Tensor): Training sequences.
    labels_training (Tensor): Training labels.
    sequences_testing (Tensor): Testing sequences.
    labels_testing (Tensor): Testing labels.
    epochs (int, optional): Number of training epochs. Default is 100.
    lowerbound (float, optional): Lower bound for early stopping condition. Default is 0.4.
    upperbound (float, optional): Upper bound for early stopping condition. Default is 0.6.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_training.parameters())

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model_training(sequences_training)

        # Reshape labels to match the output shape
        labels_training = labels_training.view(-1, 1)

        loss = criterion(output, labels_training)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            output = model_training(sequences_testing)

            # Reshape your labels to match the output shape
            labels_testing = labels_testing.view(-1, 1)

            loss = criterion(output, labels_testing)
            print('Validation Loss: ', loss.item())
        
        # Early stopping condition, based on empirical testing of the "sweet spot"
        if lowerbound <= loss.item() <= upperbound:
            #print("Early stopping, validation loss is between 0.4 and 0.6")
            break

#Save the model...
def save_model(saved_model,model_path):
    """
    Save a PyTorch model's state dictionary to a file.

    Parameters:
    saved_model (nn.Module): The PyTorch model to be saved.
    model_path (str): Path to the file where the model will be saved.
    """
    torch.save(saved_model.state_dict(), model_path)


#Save the vocab...
def save_vocab(vocab,vocaboutputfile):
    """
    Save a vocabulary dictionary to a file using pickle.

    Parameters:
    vocab (dict): The vocabulary dictionary to be saved.
    vocaboutputfile (str): Path to the file where the vocabulary will be saved.

    Returns:
    None
    """
    with open(vocaboutputfile, 'wb') as f:
            pickle.dump(vocab, f)
