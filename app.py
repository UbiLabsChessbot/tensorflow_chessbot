from flask import Flask, jsonify
from flask import abort
from flask import request
import tensorflow as tf
import numpy as np
import os
import glob
import argparse
import cv2
import numpy
import time
# Imports for computer vision
from PIL import Image
import scipy.signal
import helper_functions
import tensorflow_chessbot
import re, sys, time
import json
import os
from os import path
app = Flask(__name__)
app.config['UPLOAD_FOLDER']='/home/own/chessbot'

@app.route('/upload', methods=['POST'])
def upload():
    global predictor
    start = time.clock()
    fens = ""
    img = request.files['file']
    data = {'code':1,'message':123,'time':1}
    date = json.dumps(data)
    result=json.loads(date)
    filename = 'Chess.png'
    if os.path.exists(filename):
       os.remove(filename)
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    fen = predictor.makePredictionFromFile()
    end = time.clock()
    result['time']=(end - start)
    result['message']=fen
    print result
    return jsonify(result)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <body>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    '''


if __name__ == '__main__':
    predictor = tensorflow_chessbot.ChessboardPredictor()
    #app.run(host='0.0.0.0',port=8081)
    app.run(host='0.0.0.0',port=5000,debug=True)
