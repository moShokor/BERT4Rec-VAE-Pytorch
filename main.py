from dataloaders import dataloader_factory
from datasets import dataset_factory
from features_extractors import extractors_factory
from models import model_factory
from options import args
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    extractors = extractors_factory(args)
    dataset = dataset_factory(args, extractors)
    train_loader, val_loader, test_loader = dataloader_factory(args, dataset, extractors)
    model = model_factory(args, extractors)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    # test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    test_model = True
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
