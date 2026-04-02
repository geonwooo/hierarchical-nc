from .classifier_ops import *
from .pooling_ops import *
from .reshape_ops import *
from .scaling_ops import *
from .loss_ops import *
from .hier_ops import *


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
