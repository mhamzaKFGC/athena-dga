"""Train the three baselines that need training; CharBot needs none.

  python train_all.py            full training (CPU; roughly 20-40 min)
  python train_all.py --quick    smoke test of every trainer
  python train_all.py --only maskdga deepdga     train a subset

Saves weights next to each module: maskdga_substitute.pt, deepdga_ae.pt, deepdga_gan.pt, khaos_wgan.pt.
So later generation does not start from scratch.
"""
import argparse
import time

import deepdga
import khaos
import maskdga

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", nargs="*", default=None, help="maskdga deepdga khaos")
a = ap.parse_args()
q = a.quick
want = set(a.only) if a.only else {"maskdga", "deepdga", "khaos"}

if "maskdga" in want:
    print("== MaskDGA substitute (1D-CNN, filter widths 2-5)", flush=True)
    t = time.time()
    maskdga.train_substitute(quick=q)
    print(f"   done in {time.time() - t:.0f}s", flush=True)

if "deepdga" in want:
    print("== DeepDGA autoencoder (pretrain)", flush=True)
    t = time.time()
    deepdga.pretrain_ae(quick=q)
    print(f"   AE done in {time.time() - t:.0f}s", flush=True)
    print("== DeepDGA GAN (adversarial rounds)", flush=True)
    t = time.time()
    deepdga.train_gan(quick=q)
    print(f"   GAN done in {time.time() - t:.0f}s", flush=True)

if "khaos" in want:
    print("== Khaos n-gram WGAN-GP", flush=True)
    t = time.time()
    khaos.train(quick=q)
    print(f"   done in {time.time() - t:.0f}s", flush=True)

print("all requested baselines trained.", flush=True)
