from flask import Flask, render_template, request
import os
from model import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    prediction, confidence = predict_image(path)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        image_path="/" + path
    )

if __name__ == "__main__":
    app.run(debug=True)
