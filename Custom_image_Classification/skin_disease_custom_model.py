import os
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# === Load environment variables ===
load_dotenv(dotenv_path=r"C:\Users\Nizam\Desktop\azure_vision\.env")

PREDICTION_KEY = os.getenv("PREDICTION_KEY")
ENDPOINT = os.getenv("PREDICTION_ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
PUBLISHED_NAME = os.getenv("PUBLISHED_NAME")
IMAGE_PATH = r"C:\Users\Nizam\Desktop\azure_vision\5_VI-chickenpox (1).jpeg"   # path to your test image

# === Authenticate ===
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
prediction_client = CustomVisionPredictionClient(ENDPOINT, credentials)

# === Make prediction ===
with open(IMAGE_PATH, "rb") as image_data:
    results = prediction_client.classify_image(PROJECT_ID, PUBLISHED_NAME, image_data.read())

# === Display predictions ===
print(f"\nResults for: {IMAGE_PATH}")
for prediction in results.predictions:
    print(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
