"""Model shape shared by inference and training."""

# Parallel filter bank branches, Kim (2014) style: (LayerNum, region_size, num_filters).
# Each branch slides its filters along the word axis only (kernel width is always the
# full embedding dimension) and globally max-pools over word position, so branches run
# in parallel over the same embedding input rather than stacking. See layers/filter_bank.py.
FILTER_BANK_CONFIG = [(1, 3, 100), (2, 4, 100), (3, 5, 100)]

# Hidden ReLU layer widths between the filter banks and the output layer. Kim (2014)
# uses none; 1.0 used five (256, 128, 64, 32, 16), which sat at chance loss for
# thousands of batches before learning. One layer keeps 1.0's first width and a place
# for dropout. Changing this changes the shape `wren train` seeds the database with.
HIDDEN_WIDTHS = [256]
NUM_CLASSES = 5

SENTIMENT_LABELS = {1: "Very negative", 2: "Negative", 3: "Neutral",
                    4: "Positive", 5: "Very positive"}
