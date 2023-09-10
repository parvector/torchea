


def softsel(modelu, modelx, if_both_eq="first"):
    """
    if_both_eq: "both", "first, "second"
    """
    if not (if_both_eq == "both" or if_both_eq == "first" or if_both_eq == "second"):
        raise ValueError("if_both_eq must be 'both', 'first', 'second'")

    if all([ ufitnes == xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]) and if_both_eq == "both":
        return modelu, modelx
    elif all([ ufitnes == xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]) and if_both_eq == "first":
        return modelu
    elif all([ ufitnes == xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]) and if_both_eq == "second":
        return modelx
    elif all([ ufitnes <= xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]) \
        and not all([ ufitnes == xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]):
        return modelx
    elif all([ ufitnes >= xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]) \
        and not all([ ufitnes == xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes) ]):
        return modelu
    
def hardsel(modelu, modelx):
    if all([ ufitnes > xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes)]):
        return modelu
    elif all([ ufitnes < xfitnes for ufitnes, xfitnes in zip(modelu.fitnes, modelx.fitnes)]):
        return modelx
    else:
        return False