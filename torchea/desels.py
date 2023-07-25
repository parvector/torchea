


def softsel(modelu, modelx):
    if modelu <= modelx:
        return modelu 
    else:
        return modelx
    
def hardsel(modelu, modelx):
    if modelu < modelx:
        return modelu 
    else:
        return modelx