


def softsel(modelu, modelx, if_both_eq="first"):
    """
    if_both_eq: "both", "first, "second"
    """
    if not (if_both_eq == "both" or if_both_eq == "first" or if_both_eq == "second"):
        raise ValueError("if_both_eq must be 'both', 'first', 'second'")

    if all([ ueval == xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]) and if_both_eq == "both":
        return modelu, modelx
    elif all([ ueval == xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]) and if_both_eq == "first":
        return modelu
    elif all([ ueval == xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]) and if_both_eq == "second":
        return modelx
    elif all([ ueval <= xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]) \
        and not all([ ueval == xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]):
        return modelx
    elif all([ ueval >= xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]) \
        and not all([ ueval == xeval for ueval, xeval in zip(modelu.eval, modelx.eval) ]):
        return modelu
    
def hardsel(modelu, modelx):
    if all([ ueval > xeval for ueval, xeval in zip(modelu.eval, modelx.eval)]):
        return modelu
    elif all([ ueval < xeval for ueval, xeval in zip(modelu.eval, modelx.eval)]):
        return modelx
    else:
        return False