# Adversarial-DGA baselines

Four published evasion techniques, reimplemented to compare against **Athena** on the same footing:
each exposes `generate(n, seed) -> list[str]` returning bare DNS labels, filtered for novelty exactly
as Athena's names are (not a known Majestic-200k label, not a 1-edit lookalike of one; see
`common_gen.py`). They feed straight into the project's `scorers` and the Athena-vs-classifiers script.

| Baseline | Paper | Type | Needs training | How it works |
|---|---|---|---|---|
| **CharBot** | Peck et al., IEEE Access 2019 ([arXiv:1905.01078](https://arxiv.org/abs/1905.01078)) | rule-based | no | Change 2 random characters of a benign label; keep if novel. |
| **MaskDGA** | Sidi et al., 2019 ([arXiv:1902.08909](https://arxiv.org/abs/1902.08909)) | gradient perturbation | substitute CNN | Rewrite ~half of a real DGA name using a substitute detector's Jacobian saliency map. |
| **DeepDGA** | Anderson et al., AISec 2016 ([arXiv:1610.01969](https://arxiv.org/abs/1610.01969)) | autoencoder + GAN | yes | Char autoencoder on legit domains, repurposed as a GAN whose generator maps a random seed to an Alexa-like name. |
| **Khaos** | Yun et al., IEEE TIFS 2020 | n-gram WGAN-GP | yes | Mine frequent n-grams of legit domains; a WGAN-GP recombines them into new names. |

## Run

```powershell
python train_all.py            # trains MaskDGA substitute, DeepDGA AE+GAN, Khaos WGAN-GP (CharBot needs none)
python generate_all.py --n 1000 --seed 2026    # writes out/<name>.txt for each
```

`--quick` smoke-tests every trainer in seconds. Weights are saved next to each module
(`maskdga_substitute.pt`, `deepdga_ae.pt`, `deepdga_gan.pt`, `khaos_wgan.pt`) so generation never
retrains.

## Fidelity notes (what matches the paper, what is approximated)

- **CharBot**: faithful, the paper's whole method is "randomly modify two characters of a benign
  domain," which is exactly `charbot.py`.
- **MaskDGA**: faithful mechanism. Substitute = 1-D CNN, filter widths 2/3/4/5, max-pool halving the
  length, two FC layers, trained 5 epochs / batch 128 / Adam lr 0.01 (paper Sec. 5.3). Attack = JSM
  saliency, threshold = **median** so exactly half the characters change, each once (paper Algorithm 1).
  We differentiate the benign-class logit, so the chosen symbol most increases benignness.
- **DeepDGA**: faithful architecture (embedding d=20, 20 bigram + 10 trigram conv filters, highway,
  LSTM; box-layer generator; logistic-regression discriminator on the embedding; history
  regularization; 3 adversarial rounds). The paper's exact max-over-time/filters pooling is loosely
  specified and stands in as a per-timestep linear; sizes are CPU-scaled; a STOP token is added so
  generated names end naturally. Trained on Majestic-200k rather than Alexa-1M.
- **Khaos**: faithful pipeline (n-gram mining -> tokenizer -> WGAN-GP over n-gram tokens ->
  synthesizer -> filter). The IEEE TIFS full text is not open-access, so exact layer sizes are not
  published; we use a standard token WGAN-GP (Gulrajani et al. 2017: residual-conv generator/critic,
  gradient penalty lambda=10, n_critic=5). Treat Khaos's numbers as an approximation of the paper.

All four are CPU-only and seedable, so every run is reproducible.
