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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import torchvision.transforms as transforms
import pandas as pd

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yolo
face = face_analysis()

# vggface
model = VGGFaceCNN().double()  # .to(device)
model_dict = torch.load(
    "Weights_trained/checkpoint_6.pth",
    map_location=lambda storage, loc: storage,
)
model.load_state_dict(model_dict)
print("CARICATO WEIGHTS")
x_train = torch.load("data_20_6.txt")
dataset = pd.read_csv("dataset.csv")
x_train = x_train.detach().numpy()  # x_train.cpu().detach().numpy()
print(x_train.shape)
print("CARICATO VETTORI TRAINING")
Y_train = torch.zeros((544 * 20,))
actor_index = 0
for i in range(544):
    for c in range(20):
        Y_train[actor_index] = i
        actor_index += 1
y_train = np.asarray(Y_train)
names = [line.rstrip("\n") for line in open("./actor_list.txt")]
print(len(names))
print("CARICATO LABEL TRAINING")
# knn

n_comp = 1024
pca = PCA(n_comp)
# proj_train = pca.fit_transform(x_train)
print("FIT PCA COMPLETATO")
# tsne = TSNE(2)
# tsne_train = tsne.fit_transform(proj_train)
# print("FIT TSNE COMPLETATO")

nn = NearestNeighbor(8)
# nn = KNeighborsClassifier(n_neighbors=7, weights='distance')
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
    height, width, layers = frame.shape
    # size = (width, height)
    print("Dimensione frame originale: {}".format(frame.shape))
    r = int(height / 2)
    c = int(width / 2)

    print("dim. nuove: ", (r, c))
    frame_rid = cv2.resize(frame, (c, r))
    frame_rid = cv2.cvtColor(frame_rid, cv2.COLOR_BGR2RGB)
    rapp_r = height / r
    rapp_c = width / c

    img, box, conf = face.face_detection(frame_arr=frame_rid, model="tiny")
    nearest_neighbors_images = []

    if len(box) == 0:
        return {"msg": "No face detected"}, 400
    # SOVRAPPONGO LE DETECTION ALL'IMMAGINE
    for x, y, w, h in box:
        # print(x,y,w,h)
        x = int(x * rapp_c)
        y = int(y * rapp_r)
        h = int(h * rapp_r)
        w = int(w * rapp_c)
        margin_r = int(80 * rapp_r)
        margin_c = int(30 * rapp_c)

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        actor = frame[y : y + h + margin_r, x - margin_c : x + w - margin_c]
        print("dim. attore: ", actor.shape)
        # PROCESSING
        actor = cv2.resize(actor, (224, 224))
        norm = transforms.Normalize((0.5,), (0.5,))
        """kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        alpha = 1.5 #contrasto
        beta = 10  #luminosità
        actor = cv2.convertScaleAbs(actor, alpha=alpha, beta=beta)
        actor = cv2.filter2D(actor, -1, kernel)
        #actor = cv2.fastNlMeansDenoisingColored(actor,None,10,10,7,21)"""
        actor = cv2.cvtColor(actor, cv2.COLOR_BGR2RGB)

        actor = (
            torch.Tensor(actor)
            .permute(2, 0, 1)
            .view(1, 3, 224, 224)
            .double()
            .to(device)
        )
        actor = norm(actor)

        # pendo solo l'output del layer fc6
        _, actor = model(actor)
        actor = actor.detach().numpy()
        print("actor:", actor.shape)
        # actor = pca.transform(actor)
        # actor = actor[:, 2]

        predictions, nearest_neighbors, nearest_neighbors_indexes = nn.predict(actor)
        print("knn predict:", predictions)
        print(nearest_neighbors)
        # predictions = F.softmax(model(actor), dim=1)
        # score, index = predictions.max(-1)
        index = int(predictions)
        # nome = "{},prob. {}".format(names[index], score.item())
        nome = "{}".format(names[index])
        print(nome)

        sure = 0
        for i in range(len(nearest_neighbors)):
            if nearest_neighbors[i] == predictions:
                sure += 1
        sure /= len(nearest_neighbors)

        color = (0, 255, 0)
        if sure < 0.3:
            color = (0, 255, 255)
        if sure < 0.1:
            color = (0, 0, 255)

        cv2.rectangle(
            frame, (x - margin_c, y), (x + w - margin_c, y + h + margin_r), color, 4
        )
        cv2.putText(
            frame,
            nome,
            (x + 20, y + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        nearest_neighbors_single = []
        for i in range(len(nearest_neighbors_indexes)):
            elem = dataset.iloc[nearest_neighbors_indexes[i]]
            val = elem.to_dict()
            static_path = "static/Dataset_nuovo/{}/{}".format(
                val["CLASS_NAME"], val["IMG_NAME"]
            )
            val["IMG_NAME"] = static_path
            nearest_neighbors_single.append(val)

        nearest_neighbors_images.append(
            {"actor": nome, "neighbors": nearest_neighbors_single}
        )

    unique_id = uuid.uuid4()
    static_path = "static/classified/{}.jpg".format(unique_id)
    cv2.imwrite(static_path, frame)  # debug
    # base64_data = base64.b64encode(frame).decode("utf-8")
    # print(base64_data)

    return {"image": static_path, "nearest_neighbors": nearest_neighbors_images}
    # return send_file(static_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run()
