from motion.detection.singlemotiondetection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2


output_frame = None
lock = threading.Lock()

app = Flask(__name__)

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")

def detect_motion(frame_count):
    global vs, output_frame, lock

    md = SingleMotionDetector(accum_weight=0.1)
    total = 0

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        timestamp = datetime.datetime.now()
        cv2.putText(frame, 
                    timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), 
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if total > frame_count:
            motion = md.detect(gray)

            if motion is not None:
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

        md.update(gray)
        total += 1

        with lock:
            output_frame = frame.copy()

def generate():
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                continue

            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encoded_image) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, 
        help="ip address of this device")
    ap.add_argument("-o", "--port", type=int, required=True, 
        help="ephemeral port number of the server (1924 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32, 
        help="# of frames used to construct the background model")

    args = vars(ap.parse_args())

    t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=True)


vs.stop()
