# spimts/helpers/selection.py
def select_core_spis(available_keys, targets):
    """
    Case-insensitive, alias-aware matching of target families
    to actual SPI keys in artifacts.
    """
    avail = list(available_keys)
    alias = {
        "SpearmanR": ["spearman"],
        "Covariance": ["cov_", "covariance"],
        "MutualInfo": ["mi_", "mutualinfo", "mutual_info"],
        "TransferEntropy": ["te_", "transferentropy", "transfer_entropy"],
        "DynamicTimeWarping": ["dtw"],
        "PairwiseDistance": ["pdist", "distance"],
    }
    out = []
    for t in targets:
        pats = alias.get(t, [t.lower()])
        hit = next((k for k in avail if k.lower() == t.lower()), None)
        if not hit:
            hit = next((k for k in avail if any(p in k.lower() for p in pats)), None)
        if hit and hit not in out:
            out.append(hit)
    return out
