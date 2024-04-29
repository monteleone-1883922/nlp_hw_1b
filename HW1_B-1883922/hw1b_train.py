import sys
from src.trainer import Trainer
from src.baseline_models import BaselineModel, BaselineSimpleModel
from src.model_frozen import HateDetectionModule as FrozenModel
from src.model_unfrozen import HateDetectionModule as UnfrozenModel
from src.dataset import HaSpeeDe_Dataset, build_dataloaders_unfrozen, build_datasets_fixed_embeddings
import random as rnd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW


# def train_baseline(device: str):
#     train_dataset, val_dataset, news_test_dataset, tweets_test_dataset = build_datasets_fixed_embeddings(device)
#
#     train_loader = train_dataset.get_dataloader(64, True)
#     val_loader = val_dataset.get_dataloader(64, True)
#     news_test_loader = news_test_dataset.get_dataloader(64, True)
#     tweets_test_loader = tweets_test_dataset.get_dataloader(64, True)
#
#     neutral_count = train_dataset.neutral_count
#     hateful_count = train_dataset.hateful_count
#
#     stratified_baseline = BaselineModel(neutral_count, hateful_count)
#     random_baseline = BaselineModel(1, 2)
#     majority_baseline = BaselineModel(0, 1)
#
#     simple_model_baseline = BaselineSimpleModel(300, 1)
#
#     trainer = Trainer(simple_model_baseline, train_loader, val_loader, None, nn.BCELoss(), device,
#                       test_dataloader=news_test_loader)
#
#     trainer.train(15, name='Simple_Model_Baseline')
#     simple_model_baseline.load_state_dict(torch.load('data/Simple_Model_Baseline.pth'))
#     print_metrics(trainer, tweets_test_loader, news_test_loader,
#                   [(stratified_baseline, 'Stratified Baseline'), (random_baseline, 'Random Baseline'),
#                    (majority_baseline, 'Majority Baseline'), (simple_model_baseline, 'Simple Model Baseline')])
#

def train_baseline(device: str):
    train_dataset, val_dataset, _, _ = build_datasets_fixed_embeddings(device)

    train_loader = train_dataset.get_dataloader(64, True)
    val_loader = val_dataset.get_dataloader(64, True)
    simple_model_baseline = BaselineSimpleModel(300, 1)

    trainer = Trainer(simple_model_baseline, train_loader, val_loader, Adam(simple_model_baseline.parameters(), lr=0.2),
                      nn.BCELoss(), device)
    trainer.train(15, name='Simple_Model_Baseline')


def set_seed(new_seed):
    np.random.seed(new_seed)
    rnd.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)  # Se stai usando GPU
    torch.backends.cudnn.deterministic = True
    return new_seed, new_seed + 1


def print_metrics(trainer, tweets_test_loader, news_test_loader, models):
    for model, name in models:
        trainer.model = model
        trainer.test_dataloader = news_test_loader
        validation_loss, precision, recall, f1, accuracy = trainer.validate()
        print()
        print(f"{name} metrics on news test set")
        print(
            f"Validation loss: {validation_loss}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}")
        trainer.test_dataloader = tweets_test_loader
        validation_loss, precision, recall, f1, accuracy = trainer.validate()
        print()
        print(f"{name} metrics on tweets test set")
        print(
            f"Validation loss: {validation_loss}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}")


def train_frozen(device: str):
    train_dataset, val_dataset, _, _ = build_datasets_fixed_embeddings(device)

    train_loader = train_dataset.get_dataloader(64, True)
    val_loader = val_dataset.get_dataloader(64, True)
    sizes = [257, 50, 100, 20, 1]
    model = FrozenModel(300, 128, sizes, dropout=0.2, lstm_layers=2)
    trainer = Trainer(model, train_loader, val_loader, AdamW(model.parameters(), lr=4e-3), nn.BCELoss(), device)
    trainer.train(23, name='Frozen_Model')


def train_unfrozen(device: str):
    train_loader, val_loader, _, _ = build_dataloaders_unfrozen(device)
    sizes = [1024, 512, 256, 1]
    model = UnfrozenModel(300, 512, sizes, dropout=0.2, lstm_layers=2, embeddings=train_loader.dataset.embedding_matrix)
    trainer = Trainer(model, train_loader, val_loader, AdamW(model.parameters(), lr=4e-3), nn.BCELoss(), device)
    trainer.train(23, name='Unfrozen_Model')


def error_device(device: str):
    if device != 'cuda' and device != 'cpu':
        print('Invalid argument, please choose between cuda and cpu')
        exit(1)


def main():
    if len(sys.argv) > 2:
        device = sys.argv[1]
        error_device(device)
        if sys.argv[2] == 'baseline':
            train_baseline(device)
        elif sys.argv[2] == 'frozen':
            train_frozen(device)
        elif sys.argv[2] == 'unfrozen':
            train_unfrozen(device)
        else:
            print('Invalid argument, please choose between baseline, frozen and unfrozen')
            exit(1)
    elif len(sys.argv) == 1:
        device = sys.argv[1]
        error_device(device)
        train_unfrozen(device)
    else:
        print('Invalid number of arguments, required at least 1 argument, please choose between cuda and cpu')
        exit(1)


if __name__ == '__main__':
    main()
