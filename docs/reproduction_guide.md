# Reproduction Guide

## 1. Data

Raw MovieLens datasets are not included in this repository. Download them from the official MovieLens website.

Required datasets:

- MovieLens latest-small
- MovieLens 1M

## 2. Preprocessing

Convert ratings into a tabular format with columns:

- user_id
- item_id
- rating
- timestamp

Construct item descriptor matrices:

- MovieLens latest-small: tag-and-genre descriptors
- MovieLens 1M: genre-only descriptors

## 3. Splits

Use fixed seed-specific train-validation-test split registries.

Primary dataset seeds:

- 42
- 123
- 2026
- 7
- 99

Secondary dataset seeds:

- 42
- 123
- 2026

## 4. Final evaluation

MovieLens latest-small:

- Item-CF
- Biased MF
- Plain NMF
- Coupled NMF

MovieLens 1M:

- Biased MF
- Plain NMF
- Tuned Coupled NMF

Final MovieLens 1M tuned configuration:

- k = 60
- alpha = 0.05
- beta = 0.0001
- lambda_reg = 0.01
- epochs = 20

## 5. Outputs

Final result CSVs are stored in `results/csv/`.

Manuscript-ready LaTeX tables are stored in:

- `results/tables/`
- `paper_assets/tables/`

Manuscript-ready figures are stored in:

- `results/figures/`
- `paper_assets/figures/`

## 6. Statistical tests

Paired seed-level tests were performed using:

- paired t-test
- Wilcoxon signed-rank test
- Cohen's paired effect size dz

See:

- `results/csv/ml_latest_small_significance_tests.csv`
- `results/csv/ml1m_significance_tests.csv`
- `results/csv/statistical_tests_all_datasets.csv`
