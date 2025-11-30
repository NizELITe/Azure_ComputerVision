import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ✅ Set environment variables in code (for testing/dev)
os.environ["VISION_ENDPOINT"] = ""
os.environ["VISION_KEY"] = ""  

# Load values from environment
endpoint = os.environ["VISION_ENDPOINT"]
key = os.environ["VISION_KEY"]

# Initialize client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Path to image
image_path = r"C:\Users\Nizam\Desktop\azure_vision\imageforcaption.jpeg"

# Read image and analyze
with open(image_path, "rb") as f:
    image_data = f.read()

result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.CAPTION],
    gender_neutral_caption=True
)

# Output the result
print("Caption:", result.caption.text)
print("Confidence:", f"{result.caption.confidence:.2f}")
