import sqlite3
import ast
from types import SimpleNamespace
import json


def fetch_kernel(LayerNum):
    conn2 = sqlite3.connect("convolution_layers.db")
    kernel_cursor = conn2.cursor()

    kernel_cursor.execute("SELECT kernel FROM kernels WHERE LayerNum = ?", (LayerNum,))
    kernel = ast.literal_eval(kernel_cursor.fetchone()[0])

    conn2.commit()
    conn2.close()

    return kernel


def fetch_layer(LayerNum):
    """ Collects the weights and biases for a given layer from an external database then encodes them in an array """

    conn1 = sqlite3.connect("neuron_weights.db")
    neural_cursor = conn1.cursor()

    neural_cursor.execute("SELECT weights FROM weights WHERE LayerNum = ?", (LayerNum,))
    weights = ast.literal_eval(neural_cursor.fetchone()[0])

    neural_cursor.execute("SELECT biases FROM weights WHERE LayerNum = ?", (LayerNum,))
    biases = ast.literal_eval(neural_cursor.fetchone()[0])

    conn1.commit()
    conn1.close()

    layerValues = [weights, biases]
    return layerValues


def format_data(embeddings):

    conn3 = sqlite3.connect('word_embeddings.db')
    curr = conn3.cursor()

    with open(embeddings, 'r', encoding='utf-8') as f:
        for line in f:
            embedding = json.loads(line, strict=False, object_hook=lambda d: SimpleNamespace(**d))
            result_word = embedding.word
            result_vector = embedding.vector
            curr.execute("INSERT INTO embeddings (word, embedding) VALUES (?, ?)", (str(result_word), str(result_vector)))

    conn3.commit()
    conn3.close()


def fetch_embedding(word):
    conn3 = sqlite3.connect('word_embeddings.db')
    curr = conn3.cursor()

    curr.execute("SELECT embedding FROM embeddings WHERE word = ?", (word,))
    result = ast.literal_eval(curr.fetchone()[0])

    conn3.commit()
    conn3.close()

    return result
