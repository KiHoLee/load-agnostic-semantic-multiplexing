"""mask_partition.py (2026-10-18) — how the learned masks divide the expanded block.

For every checkpoint data/ckpt/<name>_s<seed>.pt of a shared-block transceiver it reports, per mask,
  * own-energy share: the fraction of the mask energy m_u^2 that lies in dimensions where user u holds at
    least 90 percent of the summed mask energy of the active users (1 for a disjoint partition, about
    0.05 for independent Gaussian masks), and
  * effective width: the participation ratio (sum m_u^2)^2 / sum m_u^4 (number of dimensions that carry
    most of the mask energy).
  * transmitted-energy own share and width: the same measures for the energy each user actually sends,
    m_u^2 weighted by the per-dimension power of the common transmit projection on the test embeddings
    and normalized per user (per-user power constraint).
Writes data/extra/mask_partition.json and prints seed means (manuscript Sec. IV-C, letter Comments 1.5/2.4).
Usage: python Code/mask_partition.py
"""
import json, os, statistics as st
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_lib import embedding_cache
_E = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "data", "ckpt")
NAMES = ["shared_U4", "shared_U6", "shared_agn", "shared_agn16", "shared_U4_randmask", "shared_U4_K1"]


def stats(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, m = ck["cfg"], ck["state_dict"]["mask"]
    U = cfg["U_max"] if cfg.get("agnostic") else cfg["U"]
    e = m[:U].pow(2)
    share = e / e.sum(0, keepdim=True)
    own = [(e[u] * (share[u] >= 0.9)).sum().item() / e[u].sum().item() for u in range(U)]
    width = [(e[u].sum() ** 2 / e[u].pow(2).sum()).item() for u in range(U)]
    global _E
    if _E is None:
        _E = embedding_cache("ag_news", "bert-base-uncased", "cpu")["test"][:2000]
    sd = ck["state_dict"]
    z = F.layer_norm(F.linear(_E, sd["tx_w"], sd["tx_b"]), (sd["tx_w"].shape[0],), sd["tx_ln_w"], sd["tx_ln_b"])
    sent = e * z.pow(2).mean(0)
    sent = sent / sent.sum(1, keepdim=True)
    ss = sent / sent.sum(0, keepdim=True)
    tx_own = sum((sent[u] * (ss[u] >= 0.9)).sum().item() for u in range(U)) / U
    tx_width = (sent.sum(1) ** 2 / sent.pow(2).sum(1)).mean().item()
    return {"U": U, "own_share": sum(own) / U, "effective_width": sum(width) / U,
            "tx_own_share": tx_own, "tx_effective_width": tx_width}


def main():
    out = {}
    for n in NAMES:
        runs = {s: stats(os.path.join(CK, f"{n}_s{s}.pt")) for s in (1, 2, 3) if os.path.exists(os.path.join(CK, f"{n}_s{s}.pt"))}
        if not runs:
            continue
        out[n] = {"per_seed": runs}
        for k in ("own_share", "effective_width", "tx_own_share", "tx_effective_width"):
            out[n][k + "_mean"] = st.mean(r[k] for r in runs.values())
        print(f"{n:20s} U={next(iter(runs.values()))['U']:2d} seeds={len(runs)}  mask: own share {out[n]['own_share_mean']:.2f}, width {out[n]['effective_width_mean']:.0f}"
              f" | transmitted: own share {out[n]['tx_own_share_mean']:.2f}, width {out[n]['tx_effective_width_mean']:.0f}")
    os.makedirs(os.path.join(ROOT, "data", "extra"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "data", "extra", "mask_partition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
