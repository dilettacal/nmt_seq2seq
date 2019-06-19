from torch import nn


def init_weights_uniform(m, lowerbound, upperbound):
    for name, param in m.named_parameters():
        ### sutskever: -0.08, 0.08
        nn.init.uniform_(param.data,lowerbound, upperbound)

def init_weights_normal(m, mean, std):
    for name, param in m.named_parameters():
        ### cho: mean=0, std=0.01
        nn.init.normal_(param.data, mean=mean, std=std)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if m.__class__.__name__ == "ContextSeq2Seq":
        init_weights_normal(m, mean=0, std=0.01)
    elif m.__class__.__name__ == "StandardSeq2Seq":
        init_weights_uniform(m, -0.08, 0.08)