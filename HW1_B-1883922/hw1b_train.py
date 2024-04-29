import sys
from src.trainer import Trainer
from src.baseline_models import BaselineModel, BaselineSimpleModel
from src.model_frozen import HateDetectionModule as FrozenModel
from src.model_unfrozen import HateDetectionModule as UnfrozenModel
from src.dataset import HaSpeeDe_Dataset, build_dataloaders_unfrozen, build_dataloaders_fixed_embeddings
import random as rnd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
import json


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
    train_loader, val_loader, _, _ = build_dataloaders_fixed_embeddings(device)

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


def train_frozen(device: str):
    set_seed(108)
    train_loader, val_loader, _, _ = build_dataloaders_fixed_embeddings(device)

    sizes = [257, 50, 100, 20, 1]
    model = FrozenModel(300, 128, sizes, dropout=0.2, lstm_layers=2)
    trainer = Trainer(model, train_loader, val_loader, AdamW(model.parameters(), lr=4e-3), nn.BCELoss(), device)
    trainer.train(23, name='Frozen_Model')


def train_unfrozen(device: str):
    set_seed(59)
    train_loader, val_loader, _, _ = build_dataloaders_unfrozen(device)
    save_encodings(train_loader)
    sizes = [1024, 512, 256, 1]
    model = UnfrozenModel(300, 512, sizes, dropout=0.2, lstm_layers=2, embeddings=train_loader.dataset.embeddings)
    trainer = Trainer(model, train_loader, val_loader, AdamW(model.parameters(), lr=4e-3), nn.BCELoss(), device)
    trainer.train(23, name='Unfrozen_Model')


def save_encodings(train_loader):
    with open('data/encoding.json', 'w', encoding='utf-8') as f:
        json.dump(train_loader.dataset.encoding, f)


def error_device(device: str):
    if device != 'cuda' and device != 'cpu':
        print('Invalid argument, please choose between cuda and cpu')
        exit(1)


def main():
    if len(sys.argv) > 1:
        device = sys.argv[1]
        error_device(device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('starting training baseline simple model')
    train_baseline(device)
    print()
    print('starting training unfrozen embeddings model')
    train_unfrozen(device)
    print()
    print('starting training frozen embeddings model')
    train_frozen(device)


if __name__ == '__main__':
    main()
