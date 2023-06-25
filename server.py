"""
Load the trained network, extract each frame from a video, perform face detection with yoloface
pass each face detected to the network and draw a rectangle around the face with the prediction of the actor,
in the end reconstruct the video.
"""
from flask import Flask, send_file, render_template, request
import torch
import cv2
import torch.nn.functional as F
from lib import VGGFaceCNN
import numpy as np
from yoloface import face_analysis
import base64
import uuid
from utils import NearestNeighbor

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yolo
face = face_analysis()

# vggface
model = VGGFaceCNN().double()  # .to(device)
model_dict = torch.load(
    "Weights_trained/vggface_triplet_trained_5.pth",
    map_location=lambda storage, loc: storage,
)
model.load_state_dict(model_dict)
print("CARICATO WEIGHTS")
x_train = torch.load("data_triplet.txt")
x_train = x_train.detach().numpy()  # x_train.cpu().detach().numpy()
print(x_train.shape)
print("CARICATO VETTORI TRAINING")
Y_train = torch.zeros((544 * 5,))
actor_index = 0
for i in range(544):
    for c in range(5):
        Y_train[actor_index] = i
        actor_index += 1
y_train = np.asarray(Y_train)
names = [line.rstrip("\n") for line in open("./actor_list.txt")]
print(len(names))
print("CARICATO LABEL TRAINING")
# knn
nn = NearestNeighbor()
nn.fit(x_train, y_train)


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
    if len(box) == 0:
        return {"msg": "No face detected"}, 400
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

        # pendo solo l'output del layer fc6
        _, actor = model(actor)
        actor = actor.detach().numpy()
        print("actor:", actor.shape)
        predictions, nearest_neighbors = nn.predict(actor)
        print("knn predict:", predictions, nearest_neighbors)
        # predictions = F.softmax(model(actor), dim=1)
        # score, index = predictions.max(-1)
        index = int(predictions[0])
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
    app.run()
