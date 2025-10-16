"""
Moduli classificatore CIFAR-10
"""

from .data_loader import DatasetManager, prepare_cifar10_data
from .network import ImageClassifier, create_cifar10_classifier
from .trainer import ModelTrainer, train_and_evaluate_model

__all__ = [
    'DatasetManager',
    'prepare_cifar10_data',
    'ImageClassifier',
    'create_cifar10_classifier',
    'ModelTrainer',
    'train_and_evaluate_model'
]

