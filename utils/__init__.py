from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import argparse
import random
import torch

random.seed(42)

def get_normal_novelty_classes(classes: list, split: float) -> tuple:
    random.shuffle(classes)

    split_at = int(len(classes) * split)

    normal_classes = sorted(classes[:split_at])
    novelty_classes = sorted(classes[split_at:])

    print(f"""[INFO] randomly picked
    Normal classes: {normal_classes}
    Novelty classes: {novelty_classes}""")

    return normal_classes, novelty_classes

def classes_to_integers(classes) -> list:
    return [int(class_name.split()[0]) for class_name in classes]

def classes_to_indexes(classes, targets):
    return [classes.index(target) for target in targets]
