from flask import Flask, send_file, render_template, request

app = Flask(__name__)
"""
Load the trained network, extract each frame from a video, perform face detection with yoloface
pass each face detected to the network and draw a rectangle around the face with the prediction of the actor,
in the end reconstruct the video.
"""
import torch
import cv2
import torch.nn.functional as F
from lib import VGGFace
import numpy as np
from yoloface import face_analysis
import base64
import uuid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGFace().double().to(device)
model_dict = torch.load(
    "Weights_trained/vggface_trained.pth", map_location=lambda storage, loc: storage
)
model.load_state_dict(model_dict)
names = [line.rstrip("\n") for line in open("./data/names.txt")]
print(len(names))
face = face_analysis()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    # base64_data = request.form["image"]
    # image_data = base64.b64decode(base64_data)
    if "image" not in request.files:
        return "No file uploaded", 400
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400
    # Save the uploaded file to a desired location
    frame_path = "static/" + file.filename
    file.save(frame_path)
    frame = cv2.imread(frame_path)
    # frame as input
    # print(frame.shape)
    # height, width, layers = frame.shape
    # size = (width, height)

    img, box, conf = face.face_detection(frame_arr=frame, model="tiny")
    print(img.shape)
    # SOVRAPPONGO LE DETECTION ALL'IMMAGINE
    for x, y, w, h in box:
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        actor = img[y : y + h + 80, x : x + w]
        actor = cv2.resize(actor, (224, 224))
        actor = (
            torch.Tensor(actor)
            .permute(2, 0, 1)
            .view(1, 3, 224, 224)
            .double()
            .to(device)
        )
        predictions = F.softmax(model(actor), dim=1)
        score, index = predictions.max(-1)
        # nome = "{},prob. {}".format(names[index], score.item())
        nome = "{}".format(names[index])
        print(nome)
        cv2.rectangle(frame, (x, y), (x + w, y + h + 80), (0, 255, 0), 4)
        cv2.putText(
            frame,
            nome,
            (x + 20, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (36, 255, 12),
            2,
        )
    unique_id = uuid.uuid4()
    static_path = "static/classified/{}.jpg".format(unique_id)
    cv2.imwrite(static_path, frame)  # debug
    # base64_data = base64.b64encode(frame).decode("utf-8")
    # print(base64_data)
    return {"image": static_path}
    # return send_file(static_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True)
