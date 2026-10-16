"""Inference: `Analyser().predict("great product")` -> 1..5 stars."""

import numpy as np

from . import store, text
from .config import FILTER_BANK_CONFIG, SENTIMENT_LABELS
from .layers.filter_bank import FilterBankLayer
from .layers.neural import NeuralLayer

# Reviews per forward pass. Each review is a 200x300 embedding matrix, so a long
# list would not fit in memory at once.
CHUNK = 50


def forward_pass(conv_layers, dense_layers, output_layer, texts, lookup, training=False):
    """ Runs a batch of review texts through the whole network.

    `lookup(word)` returns the word's 300 vector. Returns the softmax output,
    shape (len(texts), 5). `training` applies dropout to the filter bank features
    and the dense layers. """
    conv_input = text.pad_matrix(text.return_vector_matrix_jsonl(text.format_entry_data(texts), lookup))
    conv_input = np.asarray(conv_input, dtype=float)

    # Parallel filter bank branches over the same embedding input, concatenated into
    # one feature vector -- not a stacked pipeline like the old two ConvLayers.
    activations = np.concatenate([layer.convPass(conv_input, training) for layer in conv_layers], axis=1)

    for layer in dense_layers:
        if training:
            activations = layer.batch_layer_output_masked(activations)
        else:
            activations = layer.batch_layer_output(activations)

    return output_layer.softmax(activations)[1]


class Analyser:
    """A trained Wren model. With no arguments it loads Wren 1.0 (downloading it on first use)."""

    def __init__(self, weights=None, embeddings=None):
        if weights is None or embeddings is None:
            default_weights, default_embeddings = store.ensure()
            weights = weights or default_weights
            embeddings = embeddings or default_embeddings

        with np.load(weights) as w:
            self.conv = [FilterBankLayer(n, r, f) for n, r, f in FILTER_BANK_CONFIG]
            # Counted from the file, so 1.0's six dense layers and 1.5's two both load.
            num_dense = sum(1 for key in w.files if key.startswith("dense") and key.endswith("_weights"))
            dense = [NeuralLayer(n) for n in range(1, num_dense + 1)]
            for layer in self.conv:
                layer.kernel = w[f"conv{layer.layerNum}_kernel"]
                layer.bias = w[f"conv{layer.layerNum}_bias"]
            for layer in dense:
                layer.weights = w[f"dense{layer.layerNum}_weights"]
                layer.biases = w[f"dense{layer.layerNum}_biases"]
        self.dense, self.output = dense[:-1], dense[-1]

        with np.load(embeddings) as e:
            self._vectors = dict(zip(e["words"].tolist(), e["vectors"]))
        self._zero = np.zeros(text.EMBEDDING_DIM)

    def _lookup(self, word):
        # Words with no vector are zeros, as in training.
        return self._vectors.get(word, self._zero)

    def predict_proba(self, texts):
        """Probability of each star rating: array of shape (len(texts), 5)."""
        if isinstance(texts, str):
            texts = [texts]
        chunks = [forward_pass(self.conv, self.dense, self.output, texts[i:i + CHUNK], self._lookup)
                  for i in range(0, len(texts), CHUNK)]
        return np.concatenate(chunks) if chunks else np.zeros((0, 5))

    def predict(self, texts):
        """Star rating 1 to 5. A str gives an int, a list of str gives a list of int."""
        stars = (np.argmax(self.predict_proba(texts), axis=1) + 1).tolist()
        return stars[0] if isinstance(texts, str) else stars

    @staticmethod
    def label(stars):
        return SENTIMENT_LABELS[int(stars)]
