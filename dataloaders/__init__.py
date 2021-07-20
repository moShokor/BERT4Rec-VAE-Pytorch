from datasets import dataset_factory
from .bert import BertDataloader
from .ae import AEDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    AEDataloader.code(): AEDataloader
}


def dataloader_factory(args, dataset, extractors):
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset, extractors)
    train, val, test = dataloader.get_pytorch_dataloaders()
    smap = dataloader.smap
    return train, val, test, smap
