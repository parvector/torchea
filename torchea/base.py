import torch
import time
import hashlib
from torch import nn
from enum import Enum
from datetime import datetime
from collections.abc import Iterable
from functools import partial



class BaseIndvdl(nn.ModuleList):
    def __init__(self, target_tensors="all", birthtime=datetime.now(), name=None, *args, **kwargs) -> None:
        """
        Args:
            birthtime(float): Time of birth. The default is time.time()
            name(None of string): Name of the individual. The default is sha256 from the time of birth.
        """
        super(BaseIndvdl, self).__init__()
        self.birthtime = birthtime
        if name == None:
            self.name = hashlib.sha256(str(self.birthtime).encode()).hexdigest()
        else: 
            self.name = str(name)
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
                    
    def __lt__(self, other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval < oeval:
                pass
            else:
                return False
        return True
    
    def __le__(self,other):
        count_eq = 0
        count_le = 0
        for seval, oeval in zip(self.eval, other.eval):
            if seval == oeval:
                count_eq += 1
            if seval <= oeval:
                count_le += 1
        if count_eq == len(self.eval) or count_eq < len(self.eval):
            return False
        elif count_le == len(self.eval):
            return True

    def __ne__(self, other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval != oeval:
                pass
            else:
                return False
        return True
    
    def __ge__(self, other):
        count_eq = 0
        count_ge = 0
        for seval, oeval in zip(self.eval, other.eval):
            if seval == oeval:
                count_eq += 1
            if seval >= oeval:
                count_ge += 1
        if count_eq == len(self.eval):
            return False
        elif count_ge == len(self.eval):
            return True
    
    def __gt__(self, other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval > oeval:
                pass
            else:
                return False
        return True
    
    def __eq__(self, other):
        if any([ seval == None for seval in self.eval]) or \
            any([ oeval == None for oeval in other.eval]):
            raise TypeError("'==' is not supported if self.eval of instances have NoneType.")
        
        for seval, oeval in zip(self.eval,other.eval):
            if seval == oeval:
                pass
            else:
                return False
        return True

    def __hash__(self, *args, **kwargs):
        return super().__hash__(*args, **kwargs)
    
    def eqid(self,other):
        if id(self) == id(other):
            return True
        else:
            return False
        
class BaseEA:
    def __init__(self):
        pass

    def run(self, npop=10, ngen=10):
        pass

    def register(self, name, method, *args, **kargs):
        pmethod = partial(method, *args, **kargs)
        pmethod.__name__ = name
        pmethod.__doc__ = method.__doc__

        setattr(self, name, pmethod)

    def unregister(self, name):
        delattr(self, name)