# DGD training options

Ported from [PerturbDGD](https://github.com/adriansousapoza/PerturbDGD)'s `docs/training_options.md`. Everything lives in
`src/training/trainer.py` and is configured under `training:` in `config/config.yaml`. All new options are **off by default**
(`loss_reduction: sum`, `gmm_update_between_refits: true`, `optimizer.decoder.type: adamw`), so old behaviour is unchanged.

**The evidence below was measured in PerturbDGD (GSD gene-expression data, NB likelihood, MLP decoder), not on images.**
Treat it as a starting point and re-check on image data.

## Defaults changed in `config.yaml`

| | before | now |
|---|---|---|
| representation lr (base to final) | 0.05 to 0.001 | **0.03 to 0.003** |
| representation weight decay | 0.01 | **0** |
| decoder lr (base to final) | 0.05 to 0.001 | **0.01 to 0.001** |

PerturbDGD: latent lr 0.03 beat 0.01 (AMI 0.64 vs 0.40) and 0.1 was slightly worse; latent weight decay 0 vs 0.01 was identical;
a higher decoder lr improved reconstruction but hurt latent structure.

## Options

| option | what it does |
|---|---|
| `optimizer.decoder.type: muon` | Muon on the decoder's 2-D weights, AdamW on the rest. **ConvDecoder has only one 2-D weight (the input Linear); conv kernels are 4-D and stay on AdamW**, so the effect will be much smaller than in PerturbDGD's MLP. |
| `decoder_ema.enabled/decay` | EMA of the decoder (params + BatchNorm buffers). Validation latents use it and the best checkpoint stores it. |
| `gmm_refit_until`, `gmm_update_between_refits` | Stop refitting the GMM after an epoch / hold it constant between refits. |
| `gmm_growth.enabled/schedule` | Start with `schedule[0]` components and split the widest component at each refit epoch. PerturbDGD: worse the fewer components it starts with. |
| `cluster_jump.*` | Every `interval` epochs try moving each latent into another component; keep the move only if `sq_err - lambda_gmm*log p_GMM` improves (clean, no noise). Stats in `training_results.pth["cluster_jump_history"]`. |
| `latent_walk.*` | Write noise **into** the stored latents after each epoch's step, annealed `start` to `end`. PerturbDGD: 0.2 per epoch lost the clusters, 0.05 was neutral to slightly negative. |
| `final_gmm.*` | A separate descriptive GMM (e.g. full covariance) fitted once on the final latents, saved as `best/gmm_final.pkl`; escalates `reg_covar`, then falls back to diag, and never blocks saving. |
| `loss_reduction: sum \| mean` | `mean` divides recon and GMM terms by the batch size. |

## Notes on what was not ported

PerturbDGD's NB likelihood, supervised/with-control GMM losses, PINN term and multi-label AMI/ARI are dataset-specific and not part of this port.
Muon, `decoder_ema`, cluster jumps and `final_gmm` had no effect measurements even in PerturbDGD beyond Muon on one dataset.
