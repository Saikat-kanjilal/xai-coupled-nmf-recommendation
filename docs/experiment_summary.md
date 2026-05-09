# Experiment Summary

## Primary dataset

Dataset: MovieLens latest-small

Descriptor setting: tags + genres

Seeds: 42, 123, 2026, 7, 99

Main finding:
Coupled NMF remains statistically comparable to plain NMF while providing descriptor-grounded additive explanations.

## Secondary dataset

Dataset: MovieLens 1M

Descriptor setting: genres only

Seeds: 42, 123, 2026

Main finding:
Tuned coupled NMF is weaker than plain NMF on RMSE and MAE but remains competitive on ranking metrics.

## Interpretation

The proposed model is not positioned as a universal accuracy winner. Its contribution is competitive recommendation performance with intrinsic descriptor-grounded explanations. Descriptor richness is important.
