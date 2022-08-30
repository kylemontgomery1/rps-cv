from flask import Flask, render_template, request, Response, redirect, url_for
from flask_bootstrap import Bootstrap
import cv2
from util import *

app = Flask(__name__)
Bootstrap(app)

cam = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(cam),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    app.run(debug=True)
