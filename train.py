from networks import Autoencoder
from sklearn.model_selection import train_test_split
from datasets import RegressionDataset
import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np


def generate_ae_data(num_samples):
    X = np.linspace(0, 300, num_samples)
    return X


class Model:
    def __init__(self, input_dim=1, num_layers=1, bidirectional=False, hidden_dim=64, batch_size=8, lr=0.1):
        self.lr = lr
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.filename = "LSTM_hidden_dim-{}-num_layers-{}-dir-".format(hidden_dim, num_layers, bidirectional)

    def train(self, train_iter, test_iter, reuse_model=False):
        model = Autoencoder(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers)

        if reuse_model:
            if os.path.exists(self.filename):
                try:
                    model.load_state_dict(torch.load(f=self.filename))
                    print("Retraining saved models")
                except:
                    print("The saved model is not compatible. Starting afresh.")

        # model = FCRegression(self.input_dim, self.batch_size)
        print("Model : ", model)
        print("Batch size : ", self.batch_size)
        criterion = nn.L1Loss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)

        # Training parameters
        num_epoch = 20000

        for epoch in range(num_epoch):
            epoch_loss = 0
            for i, [inputs, labels] in enumerate(train_iter):
                # print(inputs, inputs.shape)
                if inputs.shape[0] != self.batch_size: continue
                inputs = torch.tensor(inputs).float().reshape(1, self.batch_size, -1)
                labels = torch.tensor(labels).float().reshape(-1)
                output = model(inputs)
                # print("Inputs : ", inputs)
                # print("Labels : ", labels)
                # print("Outputs : ", output)

                model.zero_grad()
                loss = criterion(output, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                epoch_loss += loss

            print(epoch, "Training loss : ", "%.2f" % epoch_loss)
            if (epoch + 1) % 100 == 0:
                self.compute_loss(dataiter=test_iter, model=model, criterion=criterion)

            # Save the model every ten epochs
            if (epoch + 1) % 200 == 0:
                torch.save(model.state_dict(), f=self.filename)

    def compute_loss(self, dataiter, model, criterion):
        epoch_loss = 0
        for i, [inputs, labels] in enumerate(dataiter):
            if inputs.shape[0] != self.batch_size: continue
            inputs = torch.tensor(inputs).float().reshape(1, self.batch_size, -1)
            labels = torch.tensor(labels).float().reshape(-1, 1)
            output = model(inputs)

            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # Print epoch loss and do manual evalutation
            if i == len(dataiter) - 2:
                print("Epoch Loss : {}".format("%.2f" % epoch_loss))
                with torch.no_grad():
                    output = model(inputs)[:8]
                    output = np.round(output.data.numpy(), 2).reshape(-1)[:8]
                    labels = np.round(labels.data.numpy()[:8], 2).reshape(-1)
                    print("{}\n{}\n\n".format(labels, output))

    def predict(self):
        pass


if __name__ == '__main__':
    # Get data
    batch_size = 8
    Y = generate_ae_data(1 * batch_size)
    # print(Y)

    # train, test = train_test_split(Y, test_size=0.25, shuffle=False)

    # print(train, test)
    train_dataset = RegressionDataset(Y, Y)
    # test_dataset = RegressionDataset(test, test)

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Split data into test and train
    # Make data iterator
    learning_rate = 0.00001
    hidden_size = 512
    model = Model(lr=0.00001, hidden_dim=hidden_size)
    model.train(train_iterator, train_iterator)
