import os
import tensorflow as tf
import numpy as np

def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling
    """



    embeddings_path=config["embeddings_path"]

    f = open(embeddings_path)
    temp = f.readlines()[:config["embeddings_dictionary_size"]]
    f.close()
    embedding_matrix = np.zeros((config["embeddings_dictionary_size"], config["embeddings_vector_size"]))

    for i,line in enumerate(temp):
      embedding_vector = np.asarray(line.split()[1:], dtype='float32')
      try:
        embedding_matrix[i] = embedding_vector
      except:
        continue
    
    cnn_model = tf.keras.Sequential()

    #embedding layer FIX THIS
    cnn_model.add(tf.keras.layers.Embedding(input_length=config["padding_size"],
                                            input_dim=config["embeddings_dictionary_size"],
                                            output_dim=config["embeddings_vector_size"],
                                            trainable=True,
                                            weights=[np.array(embedding_matrix)],name="embedding"))
    #convolution1d later
    cnn_model.add(tf.keras.layers.Conv1D(filters=100, kernel_size=2,strides=1,padding='valid',activation='relu'))
    #globalmaxpool1d layer
    cnn_model.add(tf.keras.layers.GlobalMaxPool1D())
    #dense layers
    cnn_model.add(tf.keras.layers.Dense(100, activation="relu"))
    cnn_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    cnn_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving
    """

    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))
