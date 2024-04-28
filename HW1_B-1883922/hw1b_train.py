
import sys
from src.trainer import Trainer
from src.baseline_models import BaselineModel, BaselineSimpleModel
from src.model_frozen import HateDetectionModule as FrozenModel
from src.model_unfrozen import HateDetectionModule as UnfrozenModel
from src.dataset import HaSpeeDe_Dataset


def train_baseline():













def main():


    if len(sys.argv) > 1:
        if sys.argv[1] == 'baseline':
            train_baseline()
        elif sys.argv[1] == 'frozen':
            train_frozen()
        elif sys.argv[1] == 'unfrozen':
            train_unfrozen()
        else:
            print('Invalid argument, please choose between baseline, frozen and unfrozen')
            exit(1)
    else:
        train_unfrozen()






if __name__ == '__main__':
    main()