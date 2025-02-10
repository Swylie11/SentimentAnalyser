import sqlite3
import ast
from types import SimpleNamespace
import json


def fetch_kernel(LayerNum):
    """ Searches a database table for a layer and returns its respective kernel """

    # Database connection
    conn2 = sqlite3.connect("convolution_layers.db")
    kernel_cursor = conn2.cursor()

    # SQL query
    kernel_cursor.execute("SELECT kernel FROM kernels WHERE LayerNum = ?", (LayerNum,))
    kernel = ast.literal_eval(kernel_cursor.fetchone()[0])  # Array conversion

    conn2.commit()
    conn2.close()

    return kernel


def fetch_layer(LayerNum):
    """ Collects the weights and biases for a given layer from an external database then encodes them in an array """

    # Database connection
    conn1 = sqlite3.connect("neuron_weights.db")
    neural_cursor = conn1.cursor()

    # SQL queries
    neural_cursor.execute("SELECT weights FROM weights WHERE LayerNum = ?", (LayerNum,))
    weights = ast.literal_eval(neural_cursor.fetchone()[0])  # Array conversion

    neural_cursor.execute("SELECT biases FROM weights WHERE LayerNum = ?", (LayerNum,))
    biases = ast.literal_eval(neural_cursor.fetchone()[0])  # Array conversion

    conn1.commit()
    conn1.close()

    # Data encoding into a new array
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
    """ Grabs the vector representation of a word from an external database """

    # Establishing connection
    conn3 = sqlite3.connect('word_embeddings.db')
    curr = conn3.cursor()

    # Query
    curr.execute("SELECT embedding FROM embeddings WHERE word = ?", (word,))
    result = ast.literal_eval(curr.fetchone()[0])

    conn3.commit()
    conn3.close()

    return result


def update_values(layer_number, new_weights, new_biases):
    """ replaces the neural values database with new weights and biases from backpropagation """

    # Database connection
    conn1 = sqlite3.connect("neuron_weights.db")
    neural_cursor = conn1.cursor()

    # Query
    neural_cursor.execute("""UPDATE weights SET weights = ?, biases = ? WHERE LayerNum = ?""", (str(new_weights), str(new_biases), layer_number))

    # Closing connection
    conn1.commit()
    conn1.close()
