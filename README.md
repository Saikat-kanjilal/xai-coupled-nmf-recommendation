# Descriptor-Grounded Coupled NMF for Explainable Recommendation

This repository contains the source code, configuration files, processed experimental outputs, manuscript-ready tables, and figures for the study on descriptor-grounded coupled non-negative matrix factorization for explainable recommendation.

## Summary

The project evaluates a coupled NMF recommender that jointly models user-item ratings and item-side semantic descriptors. The model preserves the additive structure of NMF and supports descriptor-grounded explanations through factor-wise recommendation score decomposition.

## Datasets

The experiments use publicly available MovieLens datasets:

- MovieLens latest-small: primary dataset, using tag-and-genre descriptors.
- MovieLens 1M: secondary validation dataset, using genre-only descriptors.

Raw MovieLens files are not redistributed in this repository. Please download the datasets from the official MovieLens website and use the preprocessing scripts or documentation to reconstruct the processed inputs.

## Final experimental design

Primary dataset:

- Dataset: MovieLens latest-small
- Descriptor setting: tags + genres
- Seeds: 42, 123, 2026, 7, 99
- Models: Item-CF, biased MF, plain NMF, coupled NMF

Secondary dataset:

- Dataset: MovieLens 1M
- Descriptor setting: genres only
- Seeds: 42, 123, 2026
- Models: biased MF, plain NMF, tuned coupled NMF

Final tuned MovieLens 1M coupled NMF configuration:

- k = 60
- alpha = 0.05
- beta = 0.0001
- lambda_reg = 0.01
- epochs = 20

## Repository contents

```text
configs/                Experiment configuration files.
src/                    Source code modules.
scripts/                Reproduction and execution scripts.
results/csv/            Final result CSV files.
results/tables/         Manuscript-ready LaTeX tables.
results/figures/        Manuscript-ready figures.
paper_assets/           Tables and figures for direct manuscript upload.
docs/                   Reproduction guide and table/figure mapping.
```

## Reproducibility

The processed result tables, generated figures, and statistical significance outputs are included. Raw datasets are excluded. The repository is intended to support reproducibility of the manuscript tables and figures and to document the experimental workflow.

## Citation

Please cite the corresponding manuscript when using this code or results.

## Clean reproducibility notebook

A clean public-facing notebook is available at:

`notebooks/01_reproduce_manuscript_outputs.ipynb`

This notebook verifies and displays the final CSV outputs, LaTeX tables, figures, and statistical significance results used in the manuscript. It does not include exploratory or failed development cells.
