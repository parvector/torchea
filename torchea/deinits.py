from torch import nn



def uniform(model,a=0.0, b=1.0):
    for param in model.parameters():
        if param.target_torchea:
            nn.init.uniform_(param, a, b)

def normal(model, mean=0.0, std=1.0):
    for param in model.parameters():
        if param.target_torchea:
            nn.init.normal_(param, mean=mean, std=std)

def constant(model, val):
    for param in model.parameters():
        if param.target_torchea:
            nn.init.constant_(param, val)

def ones(model):
    for param in model.parameters():
        if param.target_torchea:
            nn.init.ones_(param)

def zeros(model):
    for param in model.parameters():
        if param.target_torchea:
            nn.init.zeros_(param)

def trunc_normal(model, mean=0.0, std=1.0, a=-2, b=2):
    for w in model.parameters():
        nn.init.trunc_normal_(w, mean, std, a, b)

def sparse(model, sparsity, std=0.1):
    for w in model.parameters():
        nn.init.sparse_(w, sparsity, std)