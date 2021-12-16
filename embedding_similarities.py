import tensorflow as tf


cos_sim = tf.keras.losses.CosineSimilarity(axis=1)
embedding_similarities = {
    'cosine': lambda x, y: -cos_sim(x, y)
}