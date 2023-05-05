import torch
import time
import hashlib
from torch import nn
from enum import Enum
from datetime import datetime
from collections.abc import Iterable




class Task(Enum):
    Max = "MAX"
    Min = "MIN"


class BaseIndvdl(nn.ModuleList):
    def __init__(self, target_tensors="all", birthtime=datetime.now(), name=None) -> None:
        """
        Args:
            birthtime(float): Time of birth. The default is time.time()
            name(None of string): Name of the individual. The default is sha256 from the time of birth.
        """
        super(BaseIndvdl, self).__init__()
        if name == None:
            self.name = hashlib.sha256(str(time).encode()).hexdigest()
        else: 
            self.name = str(name)
        self.birthtime = birthtime
        self.eval:tuple = (None,)
        self.target_tensors = target_tensors
        
    def insert(self, index: int, module: nn.Module) -> None:
        self.freeze_module(module)
        return super().insert(index, module)
    
    def append(self, module: nn.Module) -> 'ModuleList':
        self.freeze_module(module)
        return super().append(module)

    def extend(self, modules) -> 'ModuleList':
        [self.freeze_module(module) for module in modules]
        return super().extend(modules)

    def parameters_zero(self):
        for i, param in enumerate(self.parameters()):
            if self.target_tensors == "all":
                param.zero_()
            elif i in self.target_tensors:
                param.zero_()
    
    def freeze_module(self, module:nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def count_ws(self):
        """
        return count elements of model
        """
        count = 0
        for param in self.parameters():
            count += torch.tensor(param.shape).prod().item()
        return count

    def getv(self,index):
        if self.count_ws()-1 < index or index < 0:
            raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws()-1}")
        count_ws = -1
        for param in self.parameters():
            if index == 0:
                params_value = param.data.flatten()[index]
                return params_value.item()
            len_ws = torch.tensor(param.data.shape).prod().item()
            count_ws += len_ws
            if count_ws >= index:
                count_ws -= len_ws
                for i, param_value in enumerate(param.data.flatten(), start=1):
                    if count_ws+i == index:
                        return param_value


    def setv(self,index,val):
        if self.count_ws()-1 < index or index < 0:
            raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws()-1}")
        count_ws = -1
        for param in self.parameters():
            if index == 0:
                param.data.flatten()[index] = val
                return True
            len_ws = torch.tensor(param.data.shape).prod().item()
            count_ws += len_ws
            if count_ws >= index:
                count_ws -= len_ws
                for i, _ in enumerate(param.data.flatten()):
                    count_ws+=1
                    if count_ws == index:
                        param.data.flatten()[i] = val
                        return True
                    
    def __lt__(a, b):
        for aitem, bitem in zip(a.eval,b.eval):
            if aitem < bitem:
                pass
            else:
                return False
        return True
    
    def __le__(a,b):
        for aitem, bitem in zip(a.eval,b.eval):
            if aitem <= bitem:
                pass
            else:
                return False
        return True
    
    def __ne__(a,b):
        for aitem, bitem in zip(a.eval,b.eval):
            if aitem != bitem:
                pass
            else:
                return False
        return True
    
    def __ge__(a,b):
        for aitem, bitem in zip(a.eval,b.eval):
            if aitem >= bitem:
                pass
            else:
                return False
        return True
    
    def __gt__(a,b):
        for aitem, bitem in zip(a.eval,b.eval):
            if aitem > bitem:
                pass
            else:
                return False
        return True
    
    """
    def __eq__(a,b):
        for aitem, bitem in zip(a.eval,b.eval):
            if aitem == bitem:
                pass
            else:
                return False
        return True
    """