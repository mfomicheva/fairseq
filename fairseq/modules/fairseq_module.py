import torch.nn as nn


class FairseqModule(nn.Module):
    """Base class for any stand-alone module in fairseq."""

    def __init__(self):
        super().__init__()

    @property
    def apply_dropout(self):
        return self.training
