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
    def __init__(self, birthtime=datetime.now(),*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Args:
            birthtime(float): Time of birth. The default is time.time()
            name(None of string): Name of the individual. The default is sha256 from the time of birth.
        """
        self.birthtime = birthtime
        self.name = hashlib.sha256(str(self.birthtime).encode()).hexdigest()
        self.fitnes:tuple = (None,)

    def parameters_zero(self, only_targets:bool=True):
        for param in self.parameters():
            with torch.no_grad():
                if param.target_torchea == True and only_targets:
                    param.zero_()
                elif not only_targets:
                    param.zero_()

    def count_ws(self, only_targets:bool=True):
        """
        return count elements of model
        """
        count = 0
        for param in self.parameters():
            if only_targets and param.target_torchea == True:
                count += torch.tensor(param.shape).prod().item()
            elif not only_targets:
                count += torch.tensor(param.shape).prod().item()
        return count

    def getv(self,index, only_targets=True):
        if only_targets:
            if self.count_ws()-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws()-1}")
        else:
            if self.count_ws(only_targets=only_targets)-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws(only_targets=only_targets)-1}")
        
        count_ws = -1
        first_param_indx = None
        for ti, param in enumerate(self.parameters()):
            if param.target_torchea == False and only_targets:
                continue

            if first_param_indx == None:
                first_param_indx = ti
            if index == 0 and ti == first_param_indx:
                if param.target_torchea == True and only_targets:
                    params_value = param.data.flatten()[index]
                    return params_value.item()
                elif not only_targets:
                    params_value = param.data.flatten()[index]
                    return params_value.item()
                else:
                    raise IndexError("""The index refers to a tensor element that is frozen and only_targets = True or \
                                    to a tensor element that is frozen and with no argument only_targets = False. \
                                    Fix the only_targets argument or refer to a different tensor element.""")

            len_ws = torch.tensor(param.data.shape).prod().item()
            count_ws += len_ws
            if count_ws >= index:
                count_ws -= len_ws
                for ei, param_value in enumerate(param.data.flatten(), start=1):
                    if count_ws+ei == index:
                        return param_value


    def setv(self,index,val, only_targets=True):
        if only_targets:
            if self.count_ws()-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws()-1}")
        else:
            if self.count_ws(only_targets=only_targets)-1 < index or index < 0:
                raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.count_ws(only_targets=only_targets)-1}")
        
        count_ws = -1
        first_param_indx = None
        for ti, param in enumerate(self.parameters()):
            if param.target_torchea == False and only_targets:
                continue

            if first_param_indx == None:
                first_param_indx = ti
            if index == 0 and ti == first_param_indx:
                if param.target_torchea == True and only_targets:
                    param.data.flatten()[index] = val
                    return True
                elif not only_targets:
                    param.data.flatten()[index] = val
                    return True
                else:
                    raise IndexError("""The index refers to a tensor element that is frozen and only_targets = True or \
                                    to a tensor element that is frozen and with no argument only_targets = False. \
                                    Fix the only_targets argument or refer to a different tensor element.""")

            len_ws = torch.tensor(param.data.shape).prod().item()
            count_ws += len_ws
            if count_ws >= index:
                count_ws -= len_ws
                for i, _ in enumerate(param.data.flatten()):
                    count_ws+=1
                    if count_ws == index:
                        param.data.flatten()[i] = val
                        return True 
    
    def deepcopy(self):
        copy_indvd = copy.deepcopy(self)
        for copy_param, self_param in zip(copy_indvd.parameters(), self.parameters()):
            copy_param.target_torchea = self_param.target_torchea
        copy_indvd.birthtime = datetime.now()
        copy_indvd.name = hashlib.sha256(str(copy_indvd.birthtime).encode()).hexdigest()
        return copy_indvd 


class IndvdL(BaseIndvd, nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(IndvdL, self).__init__(*args, **kwargs)

    def setarget(self, mindxs:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all", None, or list of ints
        tindxs must be "all", None, or list of ints
        """
        if mindxs == "all" or tindxs == "all":
            for param in self.parameters():
                param.target_torchea = True

        if type(mindxs) == list:
            for mi, module in enumerate(self):
                if mi in mindxs:
                    for param in module.parameters():
                        param.target_torchea = True
                else:
                    for param in module.parameters():
                        param.target_torchea = False

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.target_torchea = True
                else:
                    param.target_torchea = False


    
    def untarget(self, mindxs:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all" or list of ints
        tindxs must be "all" or list of ints
        """
        if mindxs == "all" or tindxs == "all":
            for param in self.parameters():
                param.target_torchea = False

        if type(mindxs) == list:
            for mi, module in enumerate(self):
                if mi in mindxs:
                    for param in module.parameters():
                        param.target_torchea = False

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.target_torchea = False                    


class IndvdD(BaseIndvd, nn.ModuleDict):
    def __init__(self, *args, **kwargs):
        super(IndvdD, self).__init__(*args, **kwargs)

    def setarget(self, mkeys:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mindxs must be "all" or list of ints
        tindxs must be "all" or list of ints
        """
        if mkeys == "all" or tindxs == "all":
            for param in self.parameters():
                param.target_torchea = True
                setattr(param, 'target_torchea', True)

        if type(mkeys) == list:
            for mk, module in self.items():
                if mk in mkeys:
                    for param in module.parameters():
                        param.target_torchea = True
                        setattr(param, 'target_torchea', True)
                else:
                    for param in module.parameters():
                        param.target_torchea = False
                        setattr(param, 'target_torchea', False)
                        
        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.target_torchea = True
                    setattr(param, 'target_torchea', True)
                else:
                    param.target_torchea = False
                    setattr(param, 'target_torchea', False)


    
    def untarget(self, mkeys:Union[str,list, None]="all", tindxs:Union[str,list, None]=None):
        """
        mkeys must be "all" or list of keys
        tindxs must be "all" or list of ints
        """
        if mkeys == "all" or tindxs == "all":
            for param in self.parameters():
                param.target_torchea = False

        if type(mkeys) == list:
            for mk in mkeys:
                for param in self[mk].parameters():
                    param.target_torchea = False

        if type(tindxs) == list:
            for ti, param in enumerate(self.parameters()):
                if ti in tindxs:
                    param.target_torchea = False     

class BaseEA(list):
    def __init__(self, src_indvd):
        self.src_indvd = src_indvd.deepcopy()

    def gen_pop(self, npop:int):
        self.clear()
        for _ in range(npop):
            self.append( self.src_indvd.deepcopy() )

    def run(self, npop:int=10, ngen:int=10):
        pass

    def register(self, name, method, *args, **kargs):
        pmethod = partial(method, *args, **kargs)
        pmethod.__name__ = name
        pmethod.__doc__ = method.__doc__

        setattr(self, name, pmethod)

    def unregister(self, name):
        delattr(self, name)