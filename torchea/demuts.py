import copy
import torch



def mut0(model1, model2, model3, F):
    """
    Article: Storn, Rainer & Price, Kenneth. (1995). Differential Evolution: A Simple and Efficient Adaptive Scheme for Global Optimization Over Continuous Spaces. Journal of Global Optimization. 23. 
    """
    modelv = model1.deepcopy()
    modelv.parameters_zero()
    with torch.no_grad():
        for newv, v1, v2, v3 in zip(modelv.parameters(), model1.parameters(), model2.parameters(), model3.parameters()):
            newv += v1+F*(v2-v3)
    return modelv