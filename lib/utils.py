import torch
import torch.nn as nn


def save_model(model, path):
    state_dict = model.state_dict()
    torch.save(state_dict, path)


def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_pruned_model(model, path):
    # pytorch allows dimension mismatch between the declared
    # layer dimension and the actual weight matrix.
    # However, loading pruned model cannot done by
    # directly calling net.load_state_dict() as it will
    # raise an error.
    state_dict = torch.load(path)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = state_dict[name + '.weight'].clone()
            module.bias.data = state_dict[name + '.bias'].clone()
        elif isinstance(module, nn.Linear):
            module.weight.data = state_dict[name + '.weight'].clone()
            module.bias.data = state_dict[name + '.bias'].clone()
    return model