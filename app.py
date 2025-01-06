from flask import Flask, render_template, request, redirect, url_for
import os
import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.oauth2 import service_account

app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Parameter tetap
PROJECT = "97657008905"
ENDPOINT_ID = "7602895307164090368"
LOCATION = "us-central1"
CREDENTIALS_PATH = "testing-jinam-446907-fa4c2327a45b.json"

# Fungsi prediksi
def predict_image_classification_sample(filename):
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options, credentials=credentials
    )

    with open(filename, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5
    ).to_value()

    endpoint = client.endpoint_path(
        project=PROJECT, location=LOCATION, endpoint=ENDPOINT_ID
    )
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    
    predictions = []
    for prediction in response.predictions:
        predictions.append(dict(prediction))
    return predictions

# Route untuk halaman utama
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "File gambar tidak diunggah!", 400

        try:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            predictions = predict_image_classification_sample(filename=image_path)
            os.remove(image_path)

            return render_template("results.html", predictions=predictions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return str(e), 500

    return render_template("index.html")

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)
