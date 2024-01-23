from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch.nn import Flatten
from torch import nn

from sklearn.metrics import classification_report
from torch.optim import Adam

import numpy as np
import torch
import utils
import time
import os

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

class CNN():
    def __init__(self, num_classes: int):
        # model parameters
        self.INIT_LR     = 1e-3
        self.EPOCHS      = 10

        self.model       = self._LeNet(num_classes).to(device)

    def _LeNet(self, num_classes):
        return Sequential(
            Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=20, out_channels=50, kernel_size=5),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),

            Flatten(),
            Linear(in_features=800, out_features=500),
            ReLU(),

            Linear(in_features=500, out_features=num_classes),
            LogSoftmax(dim=1)
        )

    def train(self, batch_size, classes,
        train_data_loader,
        valid_data_loader,
        test_data_loader):
        print('[INFO] initializing the LeNet model...')
        train_steps = len(train_data_loader.dataset) // batch_size 
        valid_steps = len(valid_data_loader.dataset) // batch_size 

        optimiser = Adam(self.model.parameters(), lr=self.INIT_LR)
        loss_func = nn.CrossEntropyLoss()

        classes = utils.classes_to_integers(classes)

        print(f'[INFO] training the network on {{{device_name}}}...')
        start_time = time.time()

        for e in range(self.EPOCHS):
            self.model.train()

            total_train_loss = 0
            total_valid_loss = 0

            train_correct = 0
            valid_correct = 0

            for x, y in train_data_loader:
                x = x.to(device)
                y = utils.classes_to_indexes(classes, y)
                y = torch.tensor(y).to(device)

                pred = self.model(x)
                loss = loss_func(pred, y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                total_train_loss += loss
                train_correct += (pred.argmax(1) == y).sum().item()

            with torch.no_grad():
                self.model.eval()

                for x, y in valid_data_loader:
                    x = x.to(device)
                    y = utils.classes_to_indexes(classes, y)
                    y = torch.tensor(y).to(device)

                    pred = self.model(x)
                    total_valid_loss += loss_func(pred, y)

                    valid_correct += (pred.argmax(1) == y).sum().item()

            avg_train_loss = total_train_loss / train_steps
            avg_valid_loss = total_valid_loss / valid_steps

            train_correct = train_correct / len(train_data_loader.dataset)
            valid_correct = valid_correct / len(valid_data_loader.dataset)

            print('[INFO] EPOCH: {}/{}'.format(e + 1, self.EPOCHS))
            print('Train loss: {:.6f}, Train accuracy: {:.4f}'.format(avg_train_loss, train_correct))
            print('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(avg_valid_loss, valid_correct))

        end_time = time.time()
        print('[INFO] total time taken to train the model: {:.2f}s'.format(end_time - start_time))

    def evaluate(self, classes, test_data_loader):
        classes_integers = utils.classes_to_integers(classes)

        print('[INFO] evaluating network...')
        with torch.no_grad():
            self.model.eval()

            preds = []
            targets = []
            for x, y in test_data_loader:
                x = x.to(device)
                y = utils.classes_to_indexes(classes_integers, y)
                y = torch.tensor(y).to(device)

                pred = self.model(x)
                preds.extend(pred.argmax(axis=1).cpu().numpy())
                targets.extend(y.cpu().numpy())

        if len(set(preds)) != len(set(targets)):
            print('[ERROR] your loaded model is not compatible with the chosen hyper-parameters and is likely outdated. Exiting...')
            exit()

        print(classification_report(
            y_true=targets,
            y_pred=preds,
            target_names=classes
        ))

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            print(f'[ERROR] model at path {file_path} does not exist. Exiting...')
            exit()

        print(f'[INFO] loading model from {file_path}...')
        self.model = torch.load(file_path)

    def save(self, file_path: str):
        print(f'[INFO] saving model to {file_path}...')
        parent_dir = os.path.dirname(file_path)

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        torch.save(self.model, file_path)
