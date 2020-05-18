import torch.nn as nn


class FairseqModule(nn.Module):
    """Base class for any stand-alone module in fairseq."""

    def __init__(self):
        super().__init__()
        self.apply_dropout = self.training

    @property
    def apply_dropout(self):
        return self._apply_dropout

    @apply_dropout.setter
    def apply_dropout(self, apply):
        self._apply_dropout = apply
