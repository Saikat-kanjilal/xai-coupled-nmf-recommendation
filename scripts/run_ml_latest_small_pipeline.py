"""Run or document the MovieLens latest-small experimental pipeline.

The original experiments were executed in Google Colab. This script is a
repository entry point documenting the expected pipeline order.

Recommended stages:
1. Preprocess MovieLens latest-small ratings and descriptors.
2. Generate fixed train-validation-test split registries.
3. Run baseline models and coupled NMF.
4. Generate explanation metrics.
5. Generate LaTeX tables and manuscript figures.
6. Run statistical significance tests.

See docs/reproduction_guide.md for details.
"""

def main():
    print("MovieLens latest-small pipeline entry point.")
    print("See docs/reproduction_guide.md for the Colab-based workflow.")

if __name__ == "__main__":
    main()
