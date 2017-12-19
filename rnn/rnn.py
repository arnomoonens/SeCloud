#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging

from utils import normal_init, pad_sequences

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_file(path):
    """
    Reads a csv file and makes a Numpy array with strings and labels out of it.

    Args:
        path (str): Path to the csv file to read.

    Returns:
        :obj:`numpy.Array`: Strings and labels in the dataset.
    """
    with open(path) as f:
        reader = csv.reader(f)
        data = []
        next(reader, None)
        for x in reader:
            data.append([int(x[0]), x[1]])
    return np.array(data)

def process_data(data, test_percentage):
    """
    Processes a list of labels and strings into training and test data.

    Args:
        data (:obj:`list` of :obj:`list`): Data points with for each one a label and text.
        test_percentage(float): Percentage of the data that should be used for testing.

    Returns:
        :obj:`SeCloudDataset`: Train data.
        :obj:`SeCloudDataset`: Test data.
    """
    s = set()
    for x in data:
        s = s.union(set(x[1]))
    chars_list = list(s)
    chars_list.sort()
    all_chars = "".join(chars_list)
    n_chars = len(all_chars)
    targets = []
    texts = []
    for x in data:
        targets.append(x[0])
        text = x[1]
        chars_encodings = []
        for char in text:
            arr = np.zeros((n_chars,))
            arr[chars_list.index(char)] = 1
            chars_encodings.append(arr)
        texts.append(np.array(chars_encodings))
    targets = np.array(targets, dtype="int")
    texts = np.array(texts)
    texts, lengths, indices = pad_sequences(np.array(texts), [n_chars])
    targets = targets[indices]
    np.random.shuffle(indices)
    targets = targets[indices]
    texts = texts[torch.LongTensor(indices)]
    lengths = np.array(lengths)[indices]
    n_train = round(len(texts) * (1.0 - test_percentage))
    train_data = SeCloudDataset(texts[:n_train], targets[:n_train], lengths[:n_train])
    test_data = SeCloudDataset(texts[n_train:], targets[n_train:], lengths[n_train:])
    return train_data, test_data

class SeCloudDataset(Dataset):
    """SeCloud dataset."""

    def __init__(self, texts, targets, lengths):
        self.texts = texts
        self.targets = targets
        self.lengths = lengths

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx], self.lengths[idx]

class SimpleRNN(nn.Module):
    """Wrapper around `torch.nn.LSTM`."""

    def __init__(self, n_inputs, hidden_size, n_outputs, dropout=0.0):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # self.inp = nn.Linear(n_inputs, hidden_size)
        self.rnn = nn.LSTM(n_inputs, hidden_size, num_layers=1, dropout=dropout)
        self.out = nn.Linear(hidden_size, n_outputs)
        self.softmax = nn.LogSoftmax()

        for w in self.parameters():
            w.data = normal_init(w.data.size())

    def forward(self, inp, hidden=None):
        x = inp
        # x = self.inp(x)
        x, hidden = self.rnn(x)
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x[range(len(lengths)), torch.LongTensor(lengths) - 1, :]
        x = self.out(x)
        output = self.softmax(x)
        return output, hidden

parser = argparse.ArgumentParser()
parser.add_argument("data", metavar="data", type=str, help="Path to the datafile.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train.")
parser.add_argument("--n_hidden", metavar="nhid", type=int, default=20, help="Hidden layer size.")
parser.add_argument("--learning_rate", metavar="lr", type=float, default=0.01, help="Learning rate for the optimizer.")
parser.add_argument("--dropout", type=float, default=0.0, help="If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer,")
parser.add_argument("--test", type=float, default=0.2, help="Percentage of the data to be used for testing.")
parser.add_argument("--seed", type=int, help="Seed for Numpy and PyTorch pseudo-random number generators.")
parser.add_argument("--summary", default=False, action="store_true", help="Write a summary.")
parser.add_argument("--verbose", default=False, action="store_true", help="Print info such as a progress bar.")

def main():
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    data = process_file(args.data)
    train_data, test_data = process_data(data, args.test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    rnn = SimpleRNN(len(train_data[0][0][0]), args.n_hidden, int(train_data.targets.max() + 1), dropout=args.dropout)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=args.learning_rate)
    losses = []
    accuracies = []
    if args.summary:
        writer = SummaryWriter()
    logging.getLogger().setLevel("INFO" if args.verbose else "WARNING")
    logging.info("Training for {} epochs".format(args.num_epochs))

    epoch_iterable = tqdm(range(args.num_epochs), desc="Epochs") if args.verbose else range(args.num_epochs)
    for epoch in epoch_iterable:
        batch_iterable = tqdm(train_loader, desc="Batches", leave=False) if args.verbose else train_loader
        for batch_index, (data, targets, lengths) in enumerate(batch_iterable):
            sorted_indices = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
            data = torch.nn.utils.rnn.pack_padded_sequence(
                Variable(data[torch.LongTensor(sorted_indices)].float()),
                lengths.numpy()[sorted_indices],
                batch_first=True)
            targets = Variable(targets[torch.LongTensor(sorted_indices)])
            rnn.zero_grad()
            h0 = Variable(torch.zeros(1, len(targets), args.n_hidden))
            c0 = Variable(torch.zeros(1, len(targets), args.n_hidden))
            hidden = (h0, c0)
            outputs, _ = rnn(data, hidden)
            loss = loss_function(outputs, targets)
            # if args.summary:
            #     writer.add_scalar('model/loss', loss.data[0], batch_index)
            losses.append(loss.data[0])
            predictions = outputs.data.max(dim=1)[1]
            accuracy = (predictions == targets.data).sum() / len(targets)
            accuracies.append(accuracy * 100)
            loss.backward()
            optimizer.step()
    if args.summary:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
    if args.verbose:
        logging.info("Number of train steps: {}".format(len(losses)))
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(accuracies)
        ax1.set_title("Accuracy")
        ax2.plot(losses)
        ax2.set_title("Loss")
        plt.xlabel("Train steps")
        plt.show()
    logging.info("Now testing...")

    rnn.dropout = 0.0
    test_texts = test_data.texts
    test_lengths = np.array(test_data.lengths)
    test_targets = torch.LongTensor(test_data.targets)
    sorted_indices = sorted(range(len(test_lengths)), key=lambda k: test_lengths[k], reverse=True)
    data = torch.nn.utils.rnn.pack_padded_sequence(
        Variable(test_texts[torch.LongTensor(sorted_indices)].float()),
        test_lengths[sorted_indices],
        batch_first=True)
    test_targets = Variable(test_targets[torch.LongTensor(sorted_indices)])
    rnn.zero_grad()
    h0 = Variable(torch.zeros(1, len(test_targets), args.n_hidden))
    c0 = Variable(torch.zeros(1, len(test_targets), args.n_hidden))
    hidden = (h0, c0)
    outputs, _ = rnn(data, hidden)

    predictions = outputs.data.max(dim=1)[1]
    accuracy = (predictions == test_targets.data).sum() / len(test_targets)
    if args.verbose:
        logging.info("Accuracy: {}".format(accuracy))
        y_actual = pd.Series(test_targets.data.numpy(), name="Actual")
        y_pred = pd.Series(predictions.numpy(), name="Predicted")
        confusion_matrix = pd.crosstab(y_actual, y_pred)

        f, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(confusion_matrix, square=True, annot=True)
        plt.show()
    else:
        print(accuracy)

if __name__ == '__main__':
    main()
