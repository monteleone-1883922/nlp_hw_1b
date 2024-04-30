import sys
import json
import torch
from src.trainer import Trainer
from src.baseline_models import BaselineModel, BaselineSimpleModel
from src.model_frozen import HateDetectionModule as FrozenModel
from src.model_unfrozen import HateDetectionModule as UnfrozenModel
from src.dataset import HaSpeeDe_Dataset, build_dataloaders_unfrozen, build_dataloaders_fixed_embeddings
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def print_metrics(trainer, tweets_test_loader, news_test_loader, models):
    for model, name in models:
        trainer.set_model(model)
        trainer.test_dataloader = news_test_loader
        validation_loss, precision, recall, f1, accuracy, cm = trainer.validate(test=True)
        print()
        print(f"{name} metrics on news test set")
        print(
            f"Validation loss: {validation_loss}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neutral", "hateful"])
        plt.title("Confusion matrix for " + name + " on news test set")
        disp.plot()
        plt.savefig(f"data/confusion_matrix_{name}_news.png")  # Salva la figura come un'immagine
        plt.close()
        plt.show()
        trainer.test_dataloader = tweets_test_loader
        validation_loss, precision, recall, f1, accuracy, cm = trainer.validate(test=True)
        print()
        print(f"{name} metrics on tweets test set")
        print(
            f"Validation loss: {validation_loss}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nAccuracy: {accuracy}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neutral", "hateful"])
        plt.title("Confusion matrix for " + name + " on tweets test set")
        disp.plot()
        plt.savefig(f"data/confusion_matrix_{name}_tweets.png")  # Salva la figura come un'immagine
        plt.close()
        plt.show()

def error_device(device: str):
    if device != 'cuda' and device != 'cpu':
        print('Invalid argument, please choose between cuda and cpu')
        exit(1)


def load_encodings():
    with open('data/encodings.json') as f:
        return json.load(f)


def main():
    if len(sys.argv) > 1:
        device = sys.argv[1]
        error_device(device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encodings = load_encodings()
    print("loading datasets")
    test_dataloader, _, news_test_loader, tweets_test_loader = build_dataloaders_fixed_embeddings(device)
    print("datasets loaded")
    test_dataloader.dataset.print_data_analysis(show=False)
    neutral_count = test_dataloader.dataset.neutral_count
    hateful_count = test_dataloader.dataset.hateful_count

    sizes = [1024, 512, 256, 1]
    unfrozen_model = UnfrozenModel(300, 512, sizes, lstm_layers=2, len_embeddings=21240)
    unfrozen_model.load_state_dict(torch.load('data/Unfrozen_Model.pth'))
    sizes = [257, 50, 100, 20, 1]
    frozen_model = FrozenModel(300, 128, sizes, lstm_layers=2)
    frozen_model.load_state_dict(torch.load('data/Frozen_Model.pth'))
    stratified_baseline = BaselineModel(neutral_count, hateful_count, device)
    random_baseline = BaselineModel(1, 2, device)
    majority_baseline = BaselineModel(0, 1, device)
    simple_model_baseline = BaselineSimpleModel(300, 1)
    simple_model_baseline.load_state_dict(torch.load('data/Simple_Model_Baseline.pth'))
    trainer = Trainer(None, None, None, None, nn.BCELoss(), device)
    print("begin models validation")
    print_metrics(trainer, tweets_test_loader, news_test_loader,
                  [(stratified_baseline, 'Stratified Baseline'), (random_baseline, 'Random Baseline'),
                   (majority_baseline, 'Majority Baseline'), (simple_model_baseline, 'Simple Model Baseline'),
                   (frozen_model, 'Frozen Model')])
    _, _, news_test_loader, tweets_test_loader = build_dataloaders_unfrozen(device, encodings, test_only=True)
    print_metrics(trainer, tweets_test_loader, news_test_loader,
                  [(unfrozen_model, 'Unfrozen Model')])


if __name__ == "__main__":
    main()
