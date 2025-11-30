# Import necessary libraries
import os
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

# Load keys and endpoint from .env file
load_dotenv()

PREDICTION_KEY = os.getenv("PREDICTION_KEY")
ENDPOINT = os.getenv("PREDICTION_ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")
PUBLISHED_MODEL_NAME = os.getenv("PUBLISHED_MODEL_NAME")


# Authenticate the prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
prediction_client = CustomVisionPredictionClient(endpoint=ENDPOINT, credentials=credentials)



# Path to the image file
image_path = "test_images/wheat_sample.jpg"

with open(image_path, "rb") as image_data:
    results = prediction_client.detect_image(PROJECT_ID, PUBLISHED_MODEL_NAME, image_data)



# Process and display detection results
print("\n--- Detection Results ---")
for prediction in results.predictions:
    if prediction.probability > 0.5:
        print(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
        print(f"  Bounding Box -> Left: {prediction.bounding_box.left}, Top: {prediction.bounding_box.top}, "
              f"Width: {prediction.bounding_box.width}, Height: {prediction.bounding_box.height}")
