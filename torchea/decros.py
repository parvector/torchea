import copy
import torch
from torch.distributions.uniform import Uniform



def cros0(modelx, modelv, CR=0.1, 
            distribution=Uniform(torch.tensor([0.0]), torch.tensor([1.0]))):
    """
    Article: Storn, Rainer and Price, Kenneth. Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. ournal of Global Optimization 11: 341–359, 1997.
    """
        
    modelu = modelx.deepcopy()
    modelu_count_ws = modelu.count_ws()
    for j in range(modelu_count_ws):
        if torch.rand(1).item() <= CR or j == distribution.sample().item():
            modelu.setv(j, modelv.getv(j) )
        elif torch.rand(1).item() > CR and j != distribution.sample().item():
            modelu.setv(j, modelx.getv(j) )
    return modelu