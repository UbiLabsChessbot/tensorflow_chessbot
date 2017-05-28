import tensorflow as tf
import numpy as np
import PIL.Image
import glob
import os

# Imports for visualization
import PIL.Image
from cStringIO import StringIO
import urllib2
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import scipy.signal

import helper_functions


req = urllib2.Request('http://imgur.com/u4zF5Hj.png', headers={'User-Agent' : "TensorFlow Chessbot"})
con = urllib2.urlopen(req)
image=PIL.Image.open(StringIO(con.read()))

img_arr = np.asarray(image.convert("L"), dtype=np.float32)
print(img_arr)
def getTiles(img_arr):
  """Find and slice 64 chess tiles from image in 3D Matrix"""
  # Get our grayscale image matrix
  A = tf.Variable(img_arr)


  # X & Y gradients
  Dx = gradientx(A)
  Dy = gradienty(A)


  Dx_pos = tf.clip_by_value(Dx, 0., 255., name="dx_positive")
  Dx_neg = tf.clip_by_value(Dx, -255., 0., name='dx_negative')
  Dy_pos = tf.clip_by_value(Dy, 0., 255., name="dy_positive")
  Dy_neg = tf.clip_by_value(Dy, -255., 0., name='dy_negative')

  # 1-D ampltitude of hough transform of gradients about X & Y axes
  # Chessboard lines have strong positive and negative gradients within an axis
  hough_Dx = tf.reduce_sum(Dx_pos, 0) * tf.reduce_sum(-Dx_neg, 0) / (img_arr.shape[0]*img_arr.shape[0])
  hough_Dy = tf.reduce_sum(Dy_pos, 1) * tf.reduce_sum(-Dy_neg, 1) / (img_arr.shape[1]*img_arr.shape[1])

  # Slightly closer to 3/5 threshold, since they're such strong responses
  hough_Dx_thresh = tf.reduce_max(hough_Dx) * 3/5
  hough_Dy_thresh = tf.reduce_max(hough_Dy) * 3/5

  # Transition from TensorFlow to normal values (todo, do TF right)

  # Initialize A with image array input
  # tf.initialize_all_variables().run() # will reset CNN weights so be selective

  # Local tf session
  sess = tf.Session()
  sess.run(tf.initialize_variables([A], name='getTiles_init'))

  # Get chess lines (try a fiew sets)
  hdx, hdy, hdx_thresh, hdy_thresh = sess.run(
    [hough_Dx, hough_Dy, hough_Dx_thresh, hough_Dy_thresh])
  hdx, hdy, hdx_thresh, hdy_thresh = sess.run(
    [hough_Dx, hough_Dy, hough_Dx_thresh, hough_Dy_thresh])
  lines_x, lines_y, is_match = getChessLines(hdx, hdy, hdx_thresh, hdy_thresh)
#a = np.array(image.convert("L"),dtype=np.float32)
#print(a)
#print(image.convert("L"))
