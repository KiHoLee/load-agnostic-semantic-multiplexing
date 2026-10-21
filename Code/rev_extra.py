# =========================================================
# rev_extra.py — additional evaluations requested by the 2026-10-16 audit
# of the CL2026-2366 response letter (existing checkpoints, no training).
#
#   noadapt <ckpt...>       evaluate without the one-step adaptation
#                           (adapt_steps=0) at --U loads and all SNRs
#   energy  <ckpt...>       energy-matched static / fixed allocation: an
#                           orthogonal model designed for U_d users (ortho_U4,
#                           ortho_U6) is evaluated at SNR + 10log10(U_d/U), i.e.
#                           each of U active users spends d_s/U on its own
#                           d_s/U_d dimensions (idle dimensions stay silent)
#   join    <dir>           unannounced user (2026-10-17): the system is configured for U users and
#                           one more user starts transmitting before the configuration changes.
#                           One model: the newcomer uses a free mask and the announced normalization
#                           d_s/U; the receiver decodes the announced users from their candidates.
#                           ODS: the newcomer either already uses the partition for U+1 users
#                           (misaligned transition) or reuses an existing user's block (collision).
#                           Fixed six-way: the newcomer takes a free block (no effect on the others).
#                           Reported: fidelity of the U announced users, with and without the
#                           newcomer, no adaptation step, three seeds, 2,000 frames per point.
#   diag    <ckpt...>       per-stage SIR with the signal/interference split
#                           valid for any attention: own part of z_u is
#                           sum_i a_{u,i} (|h_u| x_u (.) m_i); after the decoder
#                           each part passes the linear part of f_rx,u
#                           (layer-norm centering and gain, then W_u)
# Outputs: ../data/extra/<mode>_<name>_s<seed>.json (nothing is overwritten
# in ../data/). Usage examples are in extra_runs.sh.
# =========================================================
import argparse, json, math, os, sys
import torch
import torch.nn.functional as F

from rev_lib import (DATA, EVAL_SNRS, KAPPA, get_device, set_seed, embedding_cache, build_model,
                     sample_frames, user_gains, inner_step)

OUT = os.path.join(DATA, "extra")


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)   # torch>=2.6 defaults to weights_only=True
    cfg = ck["cfg"]
    sd = ck["state_dict"]
    n_loads = sd["mask"].shape[0] if sd["mask"].dim() == 3 else None      # load rows of a load-aware model
    model = build_model(cfg["scheme"], cfg["U_max"], cfg["d_b"], cfg["K"],
                        attention=cfg.get("attention", True),
                        per_user_rx=cfg.get("per_user_rx", False), n_loads=n_loads).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model, cfg


def adapt(model, cache, cfg, dev, snr, U, steps, n=256):
    """rev_lib.adapt with an integer seed so that non-integer SNRs work."""
    if steps == 0:
        return None
    gen = torch.Generator().manual_seed(int(cfg["seed"] + 1000 + round(snr * 100)))
    p = dict(model.params())
    model.train()
    with torch.enable_grad():
        for _ in range(steps):
            b, idx, _ = sample_frames(cache["train"], n, U, cfg["U_max"], dev, gen, cfg["agnostic"])
            g = user_gains(n, U, dev, cfg["fading"])
            p, _ = inner_step(model, p, b, idx, snr, g, cfg["alpha"], cfg["train_keys"], cfg["lam"])
    return {k: v.detach() for k, v in p.items()}


@torch.no_grad()
def evaluate(model, cache, cfg, dev, U, snrs, steps, trials=2000, bs=250, label_snrs=None, eval_seed=123):
    out = {}
    for j, snr in enumerate(snrs):
        p = adapt(model, cache, cfg, dev, snr, U, steps)
        key = label_snrs[j] if label_snrs else snr
        seed = int(eval_seed + round(snr * 100))
        set_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        model.eval()
        cos = []
        for _ in range(trials // bs):
            b, idx, _ = sample_frames(cache["test"], bs, U, cfg["U_max"], dev, gen, cfg["agnostic"])
            g = user_gains(bs, U, dev, cfg["fading"])
            cos.append(F.cosine_similarity(model(b, idx, snr, g, p=p), b, dim=-1).cpu())
        c = torch.cat(cos, 0)
        out[str(key)] = {"snr_eval": round(snr, 4), "cos": c.mean().item(), "cos_frame_std": c.mean(1).std().item(),
                         "n_frames": int(c.shape[0])}
    return out


@torch.no_grad()
def components(model, p, b, idx, gains, snr_db):
    B, U, _ = b.shape
    e = F.linear(b, p["tx_w"], p["tx_b"])
    e = F.layer_norm(e, (model.d_s,), p["tx_ln_w"], p["tx_ln_b"])
    m = p["mask"][idx]                                          # (B,U,d)
    x = e * m
    x = x * torch.sqrt(model.d_s / U / (x.pow(2).sum(-1, keepdim=True) + 1e-8))
    xg = gains.unsqueeze(-1) * x                                # (B,U,d)
    sigma = math.sqrt(1.0 / (10 ** (snr_db / 10.0)))
    n = sigma * torch.randn(B, model.d_s, device=b.device)
    y = xg.sum(1) + n
    r = y.unsqueeze(1) * m                                      # candidates r_i
    # stage 1: candidate of the own mask, r_u = y (.) m_u
    S1 = xg * m
    I1 = (xg.sum(1, keepdim=True) - xg) * m
    out = {"sir_r": (S1.pow(2).sum(-1) / I1.pow(2).sum(-1)).mean().item()}
    # attention as in the model
    if model.attention and U > 1:
        q = p["query"][idx]
        a = torch.softmax(8.0 * torch.einsum("bud,bvd->buv", F.normalize(q, dim=-1), F.normalize(r, dim=-1)), dim=-1)
    else:
        a = torch.eye(U, device=b.device).expand(B, U, U)
    out["attn_own"] = a.diagonal(dim1=1, dim2=2).mean().item()
    # stage 2: z_u = sum_i a_{u,i} r_i; target signal = sum_i a_{u,i} (xg_u (.) m_i)
    xm = xg.unsqueeze(2) * m.unsqueeze(1)                       # (B,U_src,U_cand,d): xg_w (.) m_i
    tot_c = xm.sum(1)                                           # (B,U_cand,d): all users on candidate i
    S2 = torch.einsum("bui,buid->bud", a, xm)                   # own signal on every candidate, weighted
    I2 = torch.einsum("bui,bid->bud", a, tot_c) - S2            # all other users, weighted
    out["sir_z"] = (S2.pow(2).sum(-1) / I2.pow(2).sum(-1)).mean().item()
    # stage 3: linear part of f_rx,u = W_u diag(gamma) C (layer-norm centering and gain; the common
    # 1/std scale cancels in the ratio; betas and biases are offsets and are excluded)
    def lin(zp):
        zc = zp - zp.mean(-1, keepdim=True)
        return model.decode(zc * p["rx_ln_w"], idx, {"rx_w": p["rx_w"], "rx_b": torch.zeros_like(p["rx_b"])})
    S3, I3 = lin(S2), lin(I2)
    out["sir_after_rx"] = (S3.pow(2).sum(-1) / I3.pow(2).sum(-1)).mean().item()
    # check: the parts add up to the noiseless z_u
    out["split_check"] = ((S2 + I2) - torch.einsum("bui,bid->bud", a, (xg.sum(1, keepdim=True)) * m)).abs().max().item()
    return out


# ---------- explicit transmit / receive chains for the join experiment ----------
def shared_tx(model, p, b, idx, U_norm):
    e = F.linear(b, p["tx_w"], p["tx_b"])
    e = F.layer_norm(e, (model.d_s,), p["tx_ln_w"], p["tx_ln_b"])
    x = e * p["mask"][idx]
    return x * torch.sqrt(model.d_s / U_norm / (x.pow(2).sum(-1, keepdim=True) + 1e-8))


def shared_rx(model, p, y, idx):
    m = p["mask"][idx]
    r = y.unsqueeze(1) * m
    if model.attention and idx.shape[1] > 1:
        q = p["query"][idx]
        a = torch.softmax(KAPPA * torch.einsum("bud,bvd->buv", F.normalize(q, dim=-1), F.normalize(r, dim=-1)), dim=-1)
        z = torch.einsum("buv,bvd->bud", a, r)
    else:
        z = r
    z = F.layer_norm(z, (model.d_s,), p["rx_ln_w"], p["rx_ln_b"])
    return model.decode(z, idx, p)


def ods_tx(model, p, b):
    x = F.linear(b, p["tx_w"], p["tx_b"])
    x = F.layer_norm(x, (model.d_u,), p["tx_ln_w"], p["tx_ln_b"])
    return x * torch.sqrt(model.d_u / (x.pow(2).sum(-1, keepdim=True) + 1e-8))


def ods_rx(model, p, yb):
    z = F.layer_norm(yb, (model.d_u,), p["rx_ln_w"], p["rx_ln_b"])
    return F.linear(z, p["rx_w"], p["rx_b"])


@torch.no_grad()
def join_eval(ckdir, dev, seeds=(1, 2, 3), Us=(1, 2, 3, 4, 5), snrs=(0, 10, 20, 30), trials=2000, bs=250):
    out = {"one_model": {}, "ods_new_partition": {}, "ods_collision": {}}
    cache = None
    for seed in seeds:
        agn, cfa = load(os.path.join(ckdir, f"shared_agn_s{seed}.pt"), dev)
        if cache is None:
            cache = embedding_cache(cfa["dataset"], cfa["model"], dev)
        E = cache["test"]; pa = agn.params(); agn.eval()
        ods = {U: load(os.path.join(ckdir, f"ortho_U{U}_s{seed}.pt"), dev) for U in range(1, 7)}
        for U in Us:
            for snr in snrs:
                sig = math.sqrt(1.0 / (10 ** (snr / 10.0)))
                g0 = torch.Generator().manual_seed(int(5000 + 100 * U + snr + 10000 * seed))
                acc = {k: [] for k in ("om_clean", "om_join", "om_new", "od_clean", "od_new", "od_coll", "od_newcomer")}
                for _ in range(trials // bs):
                    # ---- one model: U announced users + one newcomer on a free mask
                    b, idx, _ = sample_frames(E, bs, U + 1, cfa["U_max"], dev, g0, True)
                    g = user_gains(bs, U + 1, dev, cfa["fading"])
                    x = shared_tx(agn, pa, b, idx, U) * g.unsqueeze(-1)
                    n = sig * torch.randn(bs, agn.d_s, device=dev)
                    y_clean = x[:, :U].sum(1) + n
                    y_join = x.sum(1) + n
                    for k, y in (("om_clean", y_clean), ("om_join", y_join)):
                        acc[k].append(F.cosine_similarity(shared_rx(agn, pa, y, idx[:, :U]), b[:, :U], dim=-1).mean().item())
                    # once the receiver detects the newcomer's mask it serves it at once (no retraining, old normalization)
                    acc["om_new"].append(F.cosine_similarity(shared_rx(agn, pa, y_join, idx)[:, U], b[:, U], dim=-1).mean().item())
                    # ---- ODS configured for U users
                    mU, cU = ods[U]; pU = mU.params(); mU.eval()
                    do = mU.d_u
                    xo = ods_tx(mU, pU, b[:, :U]) * g[:, :U].unsqueeze(-1)          # (bs,U,do)
                    Y = torch.zeros(bs, agn.d_s, device=dev)
                    for u in range(U):
                        Y[:, u * do:(u + 1) * do] += xo[:, u]
                    Yn = Y + n
                    def dec(YY):
                        return torch.stack([ods_rx(mU, pU, YY[:, u * do:(u + 1) * do]) for u in range(U)], 1)
                    acc["od_clean"].append(F.cosine_similarity(dec(Yn), b[:, :U], dim=-1).mean().item())
                    # newcomer already on the partition for U+1 users (its block U of that partition)
                    m1, _ = ods[U + 1]; p1 = m1.params(); m1.eval(); dn = m1.d_u
                    xn = ods_tx(m1, p1, b[:, U:U + 1])[:, 0] * g[:, U:U + 1]
                    Y2 = Yn.clone(); Y2[:, U * dn:(U + 1) * dn] += xn
                    acc["od_new"].append(F.cosine_similarity(dec(Y2), b[:, :U], dim=-1).mean().item())
                    acc["od_newcomer"].append(F.cosine_similarity(ods_rx(m1, p1, Y2[:, U * dn:(U + 1) * dn]), b[:, U], dim=-1).mean().item())
                    # newcomer reuses the block of the last announced user (collision)
                    xc = ods_tx(mU, pU, b[:, U:U + 1])[:, 0] * g[:, U:U + 1]
                    Y3 = Yn.clone(); Y3[:, (U - 1) * do:U * do] += xc
                    acc["od_coll"].append(F.cosine_similarity(dec(Y3), b[:, :U], dim=-1).mean().item())
                key = f"U{U}_snr{snr}"
                mean = lambda k: sum(acc[k]) / len(acc[k])
                out["one_model"].setdefault(key, []).append({"seed": seed, "clean": mean("om_clean"), "join": mean("om_join"), "newcomer": mean("om_new")})
                out["ods_new_partition"].setdefault(key, []).append({"seed": seed, "clean": mean("od_clean"), "join": mean("od_new"), "newcomer": mean("od_newcomer")})
                out["ods_collision"].setdefault(key, []).append({"seed": seed, "clean": mean("od_clean"), "join": mean("od_coll")})
        print("seed", seed, "done", flush=True)
    return out


@torch.no_grad()
def join_la_eval(ckdir, name, dev, seeds=(1, 2, 3), Us=(1, 2, 3, 4, 5), snrs=(0, 10, 20, 30), trials=2000, bs=250):
    """Load-aware model: U announced users use the mask row of load U; an unannounced newcomer follows
    the same announced configuration (row U, normalization d_s/U). Reports the announced users' fidelity
    with and without the newcomer, and the newcomer's own fidelity once the receiver adds its mask."""
    out = {}
    cache = None
    for seed in seeds:
        mdl, cf = load(os.path.join(ckdir, f"{name}_s{seed}.pt"), dev)
        if cache is None:
            cache = embedding_cache(cf["dataset"], cf["model"], dev)
        E = cache["test"]; p = mdl.params(); mdl.eval()
        for U in Us:
            for snr in snrs:
                sig = math.sqrt(1.0 / (10 ** (snr / 10.0)))
                g0 = torch.Generator().manual_seed(int(7000 + 100 * U + snr + 10000 * seed))
                cl, jn, nw = [], [], []
                for _ in range(trials // bs):
                    b, idx, _ = sample_frames(E, bs, U + 1, cf["U_max"], dev, g0, True)
                    g = user_gains(bs, U + 1, dev, cf["fading"])
                    x = mdl.tx(b, idx, p, U) * g.unsqueeze(-1)            # every transmitter on the row of load U
                    n = sig * torch.randn(bs, mdl.d_s, device=dev)
                    yc, yj = x[:, :U].sum(1) + n, x.sum(1) + n
                    cl.append(F.cosine_similarity(mdl.rx(yc, idx[:, :U], p, U), b[:, :U], dim=-1).mean().item())
                    jn.append(F.cosine_similarity(mdl.rx(yj, idx[:, :U], p, U), b[:, :U], dim=-1).mean().item())
                    nw.append(F.cosine_similarity(mdl.rx(yj, idx, p, U)[:, U], b[:, U], dim=-1).mean().item())
                out.setdefault(f"U{U}_snr{snr}", []).append({"seed": seed, "clean": sum(cl) / len(cl), "join": sum(jn) / len(jn), "newcomer": sum(nw) / len(nw)})
        print("seed", seed, "done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noadapt", "energy", "diag", "join", "join_la"])
    ap.add_argument("--name", default="shared_la6")
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--U", type=int, nargs="*", default=None)
    ap.add_argument("--snr", type=float, default=10)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=2000)
    args = ap.parse_args()
    dev = get_device()
    os.makedirs(OUT, exist_ok=True)
    if args.mode == "join_la":
        R = {"mode": "join_la", "name": args.name, "device": str(dev), "result": join_la_eval(args.ckpts[0], args.name, dev, trials=args.trials)}
        json.dump(R, open(os.path.join(OUT, f"join_{args.name}.json"), "w"), indent=1); print("wrote", f"join_{args.name}.json")
        return
    if args.mode == "join":
        R = {"mode": "join", "device": str(dev), "note": "no adaptation step for every scheme; fidelity of the announced users",
             "result": join_eval(args.ckpts[0], dev, trials=args.trials)}
        json.dump(R, open(os.path.join(OUT, "join.json"), "w"), indent=1); print("wrote", os.path.join(OUT, "join.json"))
        return
    for path in args.ckpts:
        model, cfg = load(path, dev)
        cache = embedding_cache(cfg["dataset"], cfg["model"], dev)
        R = {"name": cfg["name"], "seed": cfg["seed"], "mode": args.mode, "device": str(dev)}
        if args.mode == "noadapt":
            Us = args.U or [cfg["U"]]
            R["U"] = {str(U): evaluate(model, cache, cfg, dev, U, list(EVAL_SNRS), steps=0, trials=args.trials) for U in Us}
            R["U_adapted_check"] = {str(Us[-1]): evaluate(model, cache, cfg, dev, Us[-1], list(EVAL_SNRS), steps=cfg.get("adapt_steps", 1), trials=args.trials)}
        elif args.mode == "energy":
            Ud = cfg["U"]
            Us = args.U or list(range(1, Ud + 1))
            R["design_U"] = Ud
            R["U"] = {}
            for U in Us:
                shift = 10 * math.log10(Ud / U)
                snrs = [s + shift for s in EVAL_SNRS]
                # all U_d design users are active (orthogonal: a user's fidelity does not depend on the others);
                # the shifted SNR gives each user the energy d_s/U on its d_s/U_d dimensions
                R["U"][str(U)] = {"shift_db": shift, "per_user": evaluate(model, cache, cfg, dev, Ud, snrs,
                                  steps=cfg.get("adapt_steps", 1), trials=args.trials, label_snrs=list(EVAL_SNRS))}
        else:
            U = cfg["U"]
            model.eval(); p = model.params()
            set_seed(123); gen = torch.Generator().manual_seed(123)
            b, idx, _ = sample_frames(cache["test"], args.frames, U, cfg["U_max"], dev, gen, cfg["agnostic"])
            g = user_gains(args.frames, U, dev, cfg["fading"])
            R["snr"] = args.snr
            R.update(components(model, p, b, idx, g, args.snr))
            R["cos"] = F.cosine_similarity(model(b, idx, args.snr, g), b, dim=-1).mean().item()
            for k in ("sir_r", "sir_z", "sir_after_rx"):
                R[k + "_db"] = 10 * math.log10(R[k])
        f = os.path.join(OUT, f"{args.mode}_{cfg['name']}_s{cfg['seed']}" + (f"_snr{int(args.snr)}" if args.mode == "diag" else "") + ".json")
        json.dump(R, open(f, "w"), indent=1)
        print("wrote", f, flush=True)


if __name__ == "__main__":
    main()
