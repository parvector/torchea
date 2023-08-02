import torch
import copy
import hashlib
from torch import nn
from enum import Enum
from datetime import datetime
from collections.abc import Iterable
from functools import partial
from typing import Union

class BaseIndvd:
    def __init__(self, birthtime=datetime.now(), name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Args:
            birthtime(float): Time of birth. The default is time.time()
            name(None of string): Name of the individual. The default is sha256 from the time of birth.
        """
        self.birthtime = birthtime
        if name == None:
            self.name = hashlib.sha256(str(self.birthtime).encode()).hexdigest()
        else: 
            self.name = str(name)
        self.eval:tuple = (None,)

    def parameters_zero(self, only_freeze:bool=True):
        for param in self.parameters():
            if param.requires_grad == False:
                param.zero_()

    def count_ws(self, only_freeze:bool=True):
        """
        return count elements of model
        """
        count = 0
        for param in self.parameters():
            if only_freeze and param.requires_grad == False:
                count += torch.tensor(param.shape).prod().item()
            elif not only_freeze:
                count += torch.tensor(param.shape).prod().item()
        return count

    def getv(self,index, only_freeze=True):
        if only_freeze:
            if self.count_ws()-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws()-1}")
        else:
            if self.count_ws(only_freeze=only_freeze)-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws(only_freeze=only_freeze)-1}")
        
        count_ws = -1
        first_param_indx = None
        for ti, param in enumerate(self.parameters()):
            if param.requires_grad == True and only_freeze:
                continue

            if first_param_indx == None:
                first_param_indx = ti
            if index == 0 and ti == first_param_indx:
                if param.requires_grad == False and only_freeze:
                    params_value = param.data.flatten()[index]
                    return params_value.item()
                elif not only_freeze:
                    params_value = param.data.flatten()[index]
                    return params_value.item()
                else:
                    raise IndexError("""The index refers to a tensor element that is frozen and only_freeze = True or \
                                    to a tensor element that is frozen and with no argument only_freeze = False. \
                                    Fix the only_freeze argument or refer to a different tensor element.""")

            len_ws = torch.tensor(param.data.shape).prod().item()
            count_ws += len_ws
            if count_ws >= index:
                count_ws -= len_ws
                for ei, param_value in enumerate(param.data.flatten(), start=1):
                    if count_ws+ei == index:
                        return param_value


    def setv(self,index,val, only_freeze=True):
        if only_freeze:
            if self.count_ws()-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws()-1}")
        else:
            if self.count_ws(only_freeze=only_freeze)-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws(only_freeze=only_freeze)-1}")
        
        count_ws = -1
        first_param_indx = None
        for ti, param in enumerate(self.parameters()):
            if param.requires_grad == True and only_freeze:
                continue

            if first_param_indx == None:
                first_param_indx = ti
            if index == 0 and ti == first_param_indx:
                if param.requires_grad == False and only_freeze:
                    param.data.flatten()[index] = val
                    return True
                elif not only_freeze:
                    param.data.flatten()[index] = val
                    return True
                else:
                    raise IndexError("""The index refers to a tensor element that is frozen and only_freeze = True or \
                                    to a tensor element that is frozen and with no argument only_freeze = False. \
                                    Fix the only_freeze argument or refer to a different tensor element.""")

            len_ws = torch.tensor(param.data.shape).prod().item()
            count_ws += len_ws
            if count_ws >= index:
                count_ws -= len_ws
                for i, _ in enumerate(param.data.flatten()):
                    count_ws+=1
                    if count_ws == index:
                        param.data.flatten()[i] = val
                        return True
        """
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
        """
        
                    
    def __lt__(self, other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval < oeval:
                pass
            else:
                return False
        return True
    
    def __le__(self,other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval <= oeval:
                pass
            else:
                return False
        return True

    def __ne__(self, other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval != oeval:
                pass
            else:
                return False
        return True
    
    def __ge__(self, other):
        for seval, oeval in zip(self.eval, other.eval):
            if seval >= oeval:
                pass
            else:
                return False
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


class BaseIndvdL(BaseIndvd, nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze(self, mindxs:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all", None, or list of ints
        tindxs must be "all", None, or list of ints
        """
        if mindxs == "all" or tindxs == "all":
            for param in self.parameters():
                param.requires_grad = False

        if type(mindxs) == list:
            for mi, module in enumerate(self):
                if mi in mindxs:
                    for param in module.parameters():
                        param.requires_grad = False

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.requires_grad = False


    
    def unfreeze(self, mindxs:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all" or list of ints
        tindxs must be "all" or list of ints
        """
        if mindxs == "all" or tindxs == "all":
            for param in self.parameters():
                param.requires_grad = True

        if type(mindxs) == list:
            for mi, module in enumerate(self):
                if mi in mindxs:
                    for param in module.parameters():
                        param.requires_grad = True

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.requires_grad = True     


class BaseIndvdD(BaseIndvd, nn.ModuleDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze(self, mkeys:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all" or list of ints
        tindxs must be "all" or list of ints
        """
        if mkeys == "all" or tindxs == "all":
            for param in self.parameters():
                param.requires_grad = False

        if type(mkeys) == list:
            for mk, module in self.items():
                if mk in mkeys:
                    for param in module.parameters():
                        param.requires_grad = False

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.requires_grad = False


    
    def unfreeze(self, mkeys:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all" or list of ints
        tindxs must be "all" or list of ints
        """
        if mkeys == "all" or tindxs == "all":
            for param in self.parameters():
                param.requires_grad = True

        if type(mkeys) == list:
            for mk, module in self.items():
                if mk in mkeys:
                    for param in module.parameters():
                        param.requires_grad = True

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.requires_grad = True     

class BaseEA(list):
    def __init__(self, src_indvd):
        self.src_indvd = src_indvd

    def gen_pop(self, npop:int):
        self.clear()
        for _ in range(npop):
            self.append( copy.deepcopy(self.src_indvd) )

    def run(self, npop:int=10, ngen:int=10):
        self.gen_pop(npop=npop)

    def register(self, name, method, *args, **kargs):
        pmethod = partial(method, *args, **kargs)
        pmethod.__name__ = name
        pmethod.__doc__ = method.__doc__

        setattr(self, name, pmethod)

    def unregister(self, name):
        delattr(self, name)