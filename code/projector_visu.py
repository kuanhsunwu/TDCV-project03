
"""
projector_visu.py
- Create the files needed for the tensorboard projector visualisation for Bonus task.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# path to npy file containing extracted embeddings
LOG_DIR = '.../code/tsne'

feature_vectors = np.load(os.path.join(LOG_DIR,'test_16_19000.npy'))

print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])

num_of_samples=feature_vectors.shape[0]
num_of_samples_each_class = 707

features = tf.Variable(feature_vectors, name='features')

with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'test.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_5_classes.tsv')

    # use sprite for points
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'spriteTest.png')
    embedding.sprite.single_image_dim.extend([32, 32])

    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
