from torch import nn



def uniform(model,a=0.0, b=1.0):
    for w in model.parameters():
        nn.init.uniform_(w)

def normal(model, mean=0.0, std=1.0):
    for w in model.parameters():
        nn.init.normal_(w)

def constant(model, val):
    for w in model.parameters():
        nn.init.constant_(w, val)

def ones(model):
    for w in model.parameters():
        nn.init.ones_(w)

def zeros(model):
    for w in model.parameters():
        nn.init.zeros_(w)

def trunc_normal(model, mean=0.0, std=1.0, a=-2.0, b=2.0):
    for w in model.parameters():
        nn.init.trunc_normal_(w, mean, std, a, b)

def sparse(model, sparsity, std=0.1):
    for w in model.parameters():
        nn.init.sparse_(w, sparsity, std)