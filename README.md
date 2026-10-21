# Load-Agnostic Semantic Multiplexing of BERT Embeddings in an Expanded Shared Block

Code, stored results, and trained models for the IEEE Communications Letters
paper by Ki-Ho Lee, Hyun-Ho Choi, and Jung-Ryun Lee (revised manuscript of
CL2026-2366, under review). Every figure and every number of the revised
manuscript and of the response letter can be regenerated from this repository.

## What the letter studies

Several users share one block of `d_s = K * d_b` real channel uses
(`d_b = 768`, `K = 4`, `d_s = 3072`). Each user maps its frozen BERT sentence
embedding to the block with a common learned transmit projection and its own
learned mask. The receiver applies user-wise attention and a user-specific
reverse projection (the decoder). Training is SNR-aware first-order MAML.

* The uplink is distributed: every user scales its own signal to the energy
  `d_s/U` for the announced load `U` and has its own Rayleigh coefficient with
  phase pre-compensation.
* At equal channel uses and per-user energy, the shared block matches
  orthogonal dimension sharing (ODS) and provides no gain per channel use.
* One transceiver with fixed weights, Proposed (one model), serves every load
  from one to six users and every SNR from 0 to 30 dB. It stays within 0.018
  of the transceivers trained per load and within two percent of dynamic ODS.
  Without the one-step adaptation its CosSim changes by at most 0.003.
* A user that joins before reconfiguration changes the CosSim of the announced
  users of the one-model proposal by at most 0.001, against 0.015 to 0.071
  under dynamic ODS.
* Learned masks divide the transmitted energy almost orthogonally (own-energy
  share 0.99, against 0.20 for random masks), and the user-specific decoders
  separate the users.

## Schemes

| Name in the paper | Run names | Description |
|---|---|---|
| Proposed (multi-model) | `shared_U<U>` | proposal trained separately for each load `U` |
| Proposed (one model) | `shared_agn` | one load-agnostic transceiver with six masks, trained once over random active subsets |
| Dynamic ODS (multi-model) | `ortho_U<U>` | exclusive dimensions of the same block per user, learned linear projection, retrained for each load |
| Static ODS-4 (one model) | `ortho_U4` evaluated by `rev_extra.py energy` | four fixed blocks of `d_b`, at most four users, matched energy |
| Static ODS-6 (one model) | `ortho_U6` evaluated by `rev_extra.py energy` | six fixed blocks of `d_s/6` at every load, matched energy |
| ToDMA reference | `Code/fig_cl/cl_results*.json` | token codewords, 24 slots of 128 channel uses, OMP detection |

The static schemes give each served user the energy `d_s/min(U, U_d)`. They
are evaluated as the dynamic ODS model for `U_d = 4` or `6` users at the SNR
plus `10 log10(U_d/U)` for `U <= U_d`, which equals repetition into the idle
blocks under a power limit per channel use.

## Layout

The paths are the ones cited in the manuscript and the response letter.

| Path | Content |
|---|---|
| `Code/rev_lib.py` | shared library: data split and embedding cache, transceivers, channel, MAML training, adaptation, evaluation, diagnostics |
| `Code/rev_run.py` | trains and evaluates one configuration, writes `data/<name>_s<seed>.json` and a checkpoint in `data/ckpt/` |
| `Code/rev_queue.sh` | every training run grouped by figure and table: `main`, `main_ortho`, `ortho_s3`, `scale`, `decomp`, `ablate`, `general` |
| `Code/extra_runs.sh` | evaluations added in the revision: no adaptation, matched-energy static ODS, receiver SIR, joining user, mask energy division |
| `Code/rev_extra.py` | modes `noadapt`, `energy`, `diag`, `join` used by `extra_runs.sh` |
| `Code/mask_partition.py` | own-energy share and effective width of the transmitted energy per mask |
| `Code/rev_cache.py` | builds the embedding caches `data/emb_*.pt` (not stored; AG News, SST-2, BERT-base, DistilBERT) |
| `Code/rev_collect.py` | collects all run files into `data/rev_summary.csv` and prints the tables |
| `Code/rev_replot.py` | regenerates Figs. 2 and 3 from `data/` only |
| `Code/letter_mechanism.py` | response-letter Fig. R1 and the mask statistics table |
| `Code/rev_diag.py`, `Code/rev_latency.py` | mask and attention statistics of a checkpoint; latency per frame |
| `Code/rev_pilot*.sh` | pilot runs that fixed the training length and the receiver design (`data/pilot/`) |
| `Code/bert_semcom.py`, `Code/cl_experiments.py`, `Code/cl_todma_load.py`, `Code/fig_cl/` | ToDMA reference of the original submission, reused unchanged |
| `data/*.json` | one file per run: configuration, per-SNR per-user CosSim, worst user, probe accuracy, robustness evaluations, mask statistics |
| `data/extra/` | results of `extra_runs.sh` and their printed summaries |
| `data/diag/` | mask and attention statistics of the seed-1 models |
| `data/rev_summary.csv`, `data/rev_latency.json` | summary of all runs; parameter counts and latency |
| `data/ckpt/` | seed-1 checkpoints of Proposed (multi-model) at `U = 4`, Proposed (one model), and Dynamic ODS at `U = 4` |
| `fig_cl/` | Fig. 1, Fig. 2, Fig. 3, and the letter figure R1 |

## Figure and table map

| Item | Runs | Script |
|---|---|---|
| Fig. 2 | `shared_U4_s{1,2,3}`, `shared_agn_s{1,2,3}`, `ortho_U4_s{1,2,3}`, `extra/energy_ortho_U6_s*`, `shared_U4_randmask_s1`, `shared_U4_joint_s1`, ToDMA | `rev_replot.py` |
| Fig. 3 | `shared_U{1..6}_s{1,2,3}`, `shared_agn_s{1,2,3}`, `ortho_U{1..6}_s{1,2,3}`, `extra/energy_ortho_U{4,6}_s*`, ToDMA load runs | `rev_replot.py` |
| Table II | `shared_U4_{randmask,randtx,randboth,noattn,sharedrx}_s1`, `shared_U4_K{1,2,8}_s1`, `shared_U4_K1_randmask_s1`, `shared_U6_randmask_s1`, `shared_U4_joint_s1`, `shared_U4_tr{low,high}_s1`, MAML configuration runs | `rev_collect.py` |
| Table III | robustness blocks of `shared_U4_s1` and `ortho_U4_s1`, `shared_U4_common_s1`, `ortho_U4_common_s1`, `sst2_*`, `distil_*`, `shared_U{8,12,16}_s1`, `ortho_U{8,12,16}_s1` | `rev_collect.py` |
| Sec. IV-B, one model without adaptation | `extra/noadapt_shared_agn_s*.json` | `rev_extra.py noadapt` |
| Sec. IV-B, joining user | `extra/join.json` | `rev_extra.py join` |
| Sec. IV-C, SIR per receiver stage | `extra/diag_*` | `rev_extra.py diag` |
| Sec. IV-C, energy division of the masks | `extra/mask_partition.json` | `mask_partition.py` |
| Letter Fig. R1 and mask table | `extra/diag_*`, `diag/`, `effdim_spectrum_test.npy` | `letter_mechanism.py` |

## Reproduction

Requirements: Python 3.10 or later, PyTorch, `transformers`, `datasets`,
`numpy`, `matplotlib`. Run the scripts from `Code/`.

```bash
cd Code
python rev_cache.py                                      # embedding caches
PY=python EXTRA="--per-user-rx" EP=150 BS=64 N_PAR=3 bash rev_queue.sh main
PY=python EXTRA="--per-user-rx" EP=150 BS=64 N_PAR=3 bash rev_queue.sh main_ortho   # also ortho_s3, scale, decomp, ablate, general
PY=python bash extra_runs.sh                             # evaluations added in the revision
python rev_collect.py ../data && python rev_replot.py    # tables and Figs. 2 and 3
```

`rev_replot.py`, `rev_collect.py`, and `letter_mechanism.py` read only the
stored results, so the figures and tables can be regenerated without training.
`extra_runs.sh` and `mask_partition.py` need the checkpoints of all seeds,
which `rev_queue.sh` writes to `data/ckpt/`.

Training uses 150 epochs of 100 steps with 64 frames per task, four SNR tasks
per meta-batch drawn from 0 to 25 dB, an inner rate of 5e-4, and Adam at 1e-3.
Every point is evaluated on 2,000 held-out frames, after one inner step on 256
training frames unless stated otherwise. Training seeds are 1, 2, and 3, and
the evaluation seed is 123. The runs used an Apple M5 Max (MPS, torch 2.5.1)
and an NVIDIA RTX A4500 (CUDA, torch 2.11). GPU training is not bit-exact, so
a rerun reproduces the reported values within the seed spread of the paper.

## Original submission

The code and results of the original submission, including the withdrawn
overload figure, are kept at the tag `original-submission`.

## Citation

The letter is under review. Until it appears, please cite this repository and
the earlier shared-embedding paper:

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Code and data for load-agnostic
> semantic multiplexing," 2026. [Online]. Available:
> https://github.com/KiHoLee/load-agnostic-semantic-multiplexing

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Transformer-based shared embedding
> for multiple access in semantic communications," *IEEE J. Sel. Areas
> Commun.*, vol. 44, pp. 2622-2637, 2026.
