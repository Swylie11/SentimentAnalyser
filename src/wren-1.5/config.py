"""Model shape shared by inference and training."""

# Parallel filter bank branches, Kim (2014) style: (LayerNum, region_size, num_filters).
# Each branch slides its filters along the word axis only (kernel width is always the
# full embedding dimension) and globally max-pools over word position, so branches run
# in parallel over the same embedding input rather than stacking. See layers/filter_bank.py.
FILTER_BANK_CONFIG = [(1, 3, 100), (2, 4, 100), (3, 5, 100)]

NUM_DENSE = 6  # five hidden ReLU layers plus the output layer
NUM_CLASSES = 5

SENTIMENT_LABELS = {1: "Very negative", 2: "Negative", 3: "Neutral",
                    4: "Positive", 5: "Very positive"}
