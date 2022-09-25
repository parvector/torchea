import torch, time, hashlib
from torch import nn
from enum import Enum
from collections.abc import Iterable




class Task(Enum):
    Max = "MAX"
    Min = "MIN"


class BaseIndvdl(nn.ModuleList):
    def __init__(self, target_tensors="all", birthtime=time.time(), name=None) -> None:
        """
        Args:
            target_tensors(list): The list of indexes of tensors on which evolutionary operations will be applied. If the list is "all", the operations will be performed on all tensors.
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

    def freeze_module(self, module:nn.Module):
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

    def parameters_zero(self):
        for i, params in enumerate(self.parameters()):
            if self.target_tensors == "all":
                params.zero_()
            elif i in self.target_tensors:
                params.zero_()

    def get_len(self):
        """
        return count elements of model
        """
        len = 0
        for params in list(self.parameters()):
            len += torch.tensor(params.shape).prod().item()
        return len

    def get_val(self,index):
        if self.get_len()-1 < index or index < 0:
            raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.get_len()-1}")
        count_params = 0
        for params in self.parameters():
            len_params = torch.tensor(params.data.shape).prod().item()
            if index == 0 and count_params==0:
                params_value = params.data.flatten()[index]
                return params_value.item()
            count_params += len_params
            if count_params > index:
                index_params = index-count_params
                params_value = params.data.flatten()[index_params-1]
                return params_value.item()


    def set_val(self,index,val):
        if self.get_len()-1 < index or index < 0:
            raise IndexError(f"IndexError: list index out of range. index must be >=0  and <= {self.get_len()-1}")
        count_params = 0
        for params in self.parameters():
            len_params = torch.tensor(params.data.shape).prod().item()
            count_params += len_params
            if  count_params >= index:
                index_params = index-count_params
                params.data.flatten()[index_params-1] = val
                return True
        raise IndexError(f"The value {val} is not assigned to index {index}.")
