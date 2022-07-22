import torch, time, hashlib
from torch import nn
from enum import Enum
from functools import partial
from collections import Iterable




class Task(Enum):
    Max = "MAX"
    Min = "MIN"


class BaseIndvdl(nn.ModuleList):
    def __init__(self, birthtime=time.time(), name=None) -> None:
        super(BaseIndvdl, self).__init__()
        if name == None:
            self.name = hashlib.sha256(str(time).encode()).hexdigest()
        else: 
            self.name = str(name)
        self.birthtime = birthtime
        self.eval:tuple = (None,)

    def freeze_module(self, module):
        for parameter in list(module.parameters()):
            parameter.requires_grad = False
        
    def append(self, module: nn.Module) -> 'ModuleList':
        self.freeze_module(module)
        return super().append(module)

    def insert(self, index: int, module: nn.Module) -> None:
        self.freeze_module(module)
        return super().insert(index, module)

    def extend(self, modules: Iterable[nn.Module]) -> 'ModuleList':
        self.freeze_module(modules)
        return super().extend(modules)

    def pop(self, index:int) -> 'Module':
        pop_module = self[index]
        del self[index]
        return pop_module