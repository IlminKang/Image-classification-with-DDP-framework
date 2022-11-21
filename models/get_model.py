import torchvision
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

from models.encoders import get_encoder
from models.transformer import get_object_transformer

def load_model(args, device):
    model, last_dim = get_encoder.load_encoder(args, device)

    return model



