import tensorflow as tf
import numpy as np
import PIL
import urllib, cStringIO
import glob

import helper_functions as hf
import tensorflow_chessbot

np.set_printoptions(precision=2, suppress=True)

all_paths = np.array(glob.glob("tiles/train_tiles_C/*/*.png")) # TODO : (set labels correctly)

# Shuffle order of paths so when we split the train/test sets the order of files doesn't affect it
np.random.shuffle(all_paths)

ratio = 0.9 # training / testing ratio
divider = int(len(all_paths) * ratio)
train_paths = all_paths[divider:]
test_paths = all_paths[divider:]
train_images, train_labels = hf.loadFENtiles(train_paths)
train_dataset = hf.DataSet(train_images, train_labels, dtype=tf.float32)
x = tf.placeholder(tf.float32, [None, 32*32])
W = tf.Variable(tf.zeros([32*32, 13]))
b = tf.Variable(tf.zeros([13]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 13])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)

N = 6000
print "Training for %d steps..." % N
for i in range(N):
    batch_xs, batch_ys = train_dataset.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if ((i+1) % 500) == 0:
        print "\t%d/%d" % (i+1, N)
print "Finished training."
