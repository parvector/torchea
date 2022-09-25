import copy, torch
from importlib.metadata import requires



def crosDE(modelx, modelv, CR=0.1):
    """
    Article: Storn, Rainer and Price, Kenneth. Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. ournal of Global Optimization 11: 341–359, 1997.
    """
    
    modelu = copy.deepcopy(modelx)
    modelu_len = modelx.get_len()
    for j in range(modelu_len):
        if torch.rand(1).item() <= CR and j == torch.randint(0,modelu_len,(1,)).item():
            modelu.set_val(j, modelv.get_val(j) )
        elif torch.rand(1).item() > CR and j != torch.randint(0,modelu_len,(1,)).item():
            modelu.set_val(j, modelx.get_val(j) )
    return modelu