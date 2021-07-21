from dataloaders import dataloader_factory
from datasets import dataset_factory
from features_extractors import extractors_factory
from models import model_factory
from options import args
from trainers import trainer_factory
from utils import *
import pickle5 as pickle


def prepare():
    export_root = setup_train(args)
    extractors = extractors_factory(args)
    dataset = dataset_factory(args, extractors)
    train_loader, val_loader, test_loader, smap = dataloader_factory(args, dataset, extractors)
    model = model_factory(args, extractors)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root, smap)
    return trainer


def train():
    trainer = prepare()
    trainer.train()
    trainer.test()


def test():
    trainer = prepare()
    model_path = args.test_model_path
    trainer.test(model_path)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode')
