from cnn import CNN

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import argparse
import random
import torch
import utils
import os

MEANING_OF_LIFE = 42
TRAIN_SPLIT     = 0.75
NORMAL_SPLIT    = 0.8
BATCH_SIZE      = 64
ROOT            = 'data'
MODEL_PATH      = 'output/cnn.pth'

random.seed(MEANING_OF_LIFE)

def parse_args():
    parser = argparse.ArgumentParser(
        prog='CNN static classifier based on LeNet architecture'
    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    return parser.parse_args()

def main(args):
    if os.path.exists(MODEL_PATH) and not args.test:
        print(f'[ERROR] model already exists at path {MODEL_PATH}. Exiting...')
        exit()

    print('[INFO] loading the MNIST dataset...')
    train_data = MNIST(root=ROOT, train=True, download=True, transform=ToTensor())
    test_data = MNIST(root=ROOT, train=False, download=True, transform=ToTensor())

    (normal_classes, novelty_classes) = utils.get_normal_novelty_classes(train_data.classes, NORMAL_SPLIT)
    normal_classes_int = utils.classes_to_integers(normal_classes)

    train_data = Subset(train_data, [i for i, target in enumerate(train_data.targets) if target in normal_classes_int])
    test_data = Subset(test_data, [i for i, target in enumerate(test_data.targets) if target in normal_classes_int])

    print('[INFO] generating the train/validation split...')
    num_train_samples = int(len(train_data) * TRAIN_SPLIT)
    num_valid_samples = len(train_data) - num_train_samples
    (train_data, valid_data) = random_split(train_data,
        [num_train_samples, num_valid_samples],
        generator=torch.Generator().manual_seed(MEANING_OF_LIFE)
    )

    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_data_loader  = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = CNN(len(normal_classes))
    if args.train:
        model.train(BATCH_SIZE, normal_classes, \
            train_data_loader, \
            valid_data_loader, \
            test_data_loader)
        model.save(MODEL_PATH)
    elif args.test:
        model.load(MODEL_PATH)
        model.evaluate(normal_classes, test_data_loader)
    else:
        print('[ERROR] unrecognized argument. Exiting...')
        exit()

if __name__ == '__main__':
    args = parse_args()
    main(args)
