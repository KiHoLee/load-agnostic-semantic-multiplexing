#!/usr/bin/env bash
# Evaluations added in the revision (manuscript Secs. IV-B to IV-D, response letter
# Comments 1.5, 1.6, 2.3, 2.4, 2.7). Run in Code/ after rev_queue.sh has produced the
# checkpoints in ../data/ckpt/. Outputs go to ../data/ and ../data/extra/.
set -u
PY=${PY:-python}
C=../data/ckpt
mkdir -p ../data/extra
COMMON="--scheme shared --U 6 --K 4 --agnostic --seed 1 --eval-U 1 2 3 4 5 6 --probe --per-user-rx --epochs 150 --steps 100 --batch 64 --trials 2000 --out-dir ../data"
# Proposed (one model) on SST-2 and with DistilBERT (Sec. IV-D)
$PY rev_run.py --name sst2_shared_agn   $COMMON --dataset sst2
$PY rev_run.py --name distil_shared_agn $COMMON --model distilbert-base-uncased
# Proposed (one model) without the one-step adaptation (Sec. IV-B)
$PY rev_extra.py noadapt $C/shared_agn_s1.pt $C/shared_agn_s2.pt $C/shared_agn_s3.pt --U 1 2 3 4 5 6
# Static ODS-4 and Static ODS-6 at matched per-user energy (Sec. IV-A, Figs. 2 and 3)
$PY rev_extra.py energy $C/ortho_U4_s1.pt $C/ortho_U4_s2.pt $C/ortho_U4_s3.pt $C/ortho_U6_s1.pt $C/ortho_U6_s2.pt $C/ortho_U6_s3.pt
# SIR at the receiver stages (Sec. IV-C, letter Fig. R1(b))
for snr in 10 20; do
  $PY rev_extra.py diag $C/shared_U4_s1.pt $C/shared_U4_K1_s1.pt $C/shared_U6_s1.pt \
      $C/shared_U4_randmask_s1.pt $C/shared_U4_randmask_s2.pt $C/shared_U4_randmask_s3.pt \
      $C/shared_U4_K1_randmask_s1.pt $C/shared_U6_randmask_s1.pt \
      $C/shared_U4_sharedrx_s1.pt $C/shared_U6_sharedrx_s1.pt $C/shared_U4_noattn_s1.pt $C/shared_U6_noattn_s1.pt --snr $snr
done
# A user that joins before reconfiguration (Sec. IV-B, last paragraph)
$PY rev_extra.py join $C
# Division of the transmitted energy by the masks (Sec. IV-C)
$PY mask_partition.py
# Letter figure R1 and mask table, then the summary table and Figs. 2 and 3
$PY letter_mechanism.py
$PY rev_collect.py ../data
$PY rev_replot.py
