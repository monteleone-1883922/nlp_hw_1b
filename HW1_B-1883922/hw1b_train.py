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


def train_baseline(device: str, train_loader, val_loader):

    simple_model_baseline = BaselineSimpleModel(300, 1)

    trainer = Trainer(simple_model_baseline, train_loader, val_loader, Adam(simple_model_baseline.parameters(), lr=0.2),
                      nn.BCELoss(), device)
    print('training')
    trainer.train(15, name='Simple_Model_Baseline')
    print('training finished')


def set_seed(new_seed):
    np.random.seed(new_seed)
    rnd.seed(new_seed)
    torch.manual_seed(new_seed)
    torch.cuda.manual_seed_all(new_seed)  # Se stai usando GPU
    torch.backends.cudnn.deterministic = True
    return new_seed, new_seed + 1


def train_frozen(device: str, train_loader, val_loader):
    set_seed(108)
    sizes = [257, 50, 100, 20, 1]
    model = FrozenModel(300, 128, sizes, dropout=0.2, lstm_layers=2)
    trainer = Trainer(model, train_loader, val_loader, AdamW(model.parameters(), lr=4e-3), nn.BCELoss(), device)
    print('training')
    trainer.train(23, name='Frozen_Model')
    print('training finished')


def train_unfrozen(device: str):
    set_seed(59)
    print('loading datasets')
    train_loader, val_loader, _, _ = build_dataloaders_unfrozen(device, ignore_test=True)
    print('datasets loaded')
    save_encodings(train_loader)
    sizes = [1024, 512, 256, 1]
    embeddings = train_loader.dataset.embeddings
    embeddings = sorted(list(embeddings.items()), key=lambda x: x[0])
    embeddings = torch.tensor([x[1] for x in embeddings])
    model = UnfrozenModel(300, 512, sizes, dropout=0.2, lstm_layers=2, embeddings=embeddings)
    trainer = Trainer(model, train_loader, val_loader, AdamW(model.parameters(), lr=4e-3), nn.BCELoss(), device)
    print('training')
    trainer.train(23, name='Unfrozen_Model')
    print('training finished')


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
    print('loading datasets')
    train_loader, val_loader, _, _ = build_dataloaders_fixed_embeddings(device, ignore_test=True)
    print('datasets loaded')
    print('starting training baseline simple model')
    train_baseline(device, train_loader, val_loader)
    print()
    print('starting training frozen embeddings model')
    train_frozen(device, train_loader, val_loader)
    print()
    print('starting training unfrozen embeddings model')
    train_unfrozen(device)



if __name__ == '__main__':
    main()
