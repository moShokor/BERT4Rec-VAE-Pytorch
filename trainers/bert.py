from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, smap):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, smap)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels, additional = batch[0], batch[1], batch[2:]
        logits = self.model(seqs, additional)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels, additional = batch[0], batch[1], batch[2], batch[3:]
        scores = self.model(seqs, additional)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        y, indexes = (candidates * labels).max(1)
        metrics, pure_res = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        h = {k:(res*y).tolist() for k, res in pure_res.items()}
        return metrics, h, y.tolist()
