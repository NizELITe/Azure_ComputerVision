from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import os

# ===== Load environment variables =====

load_dotenv(dotenv_path=r"C:\Users\Nizam\Desktop\azure_vision\.env")

endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_KEY")

# Create client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# ===== Image to Analyze =====
image_path = r"C:\Users\Nizam\Desktop\azure_vision\images.jpeg"

# Read image bytes
with open(image_path, "rb") as f:
    image_data = f.read()

print(f"\nAnalyzing text from: {image_path}\n")

# ===== Analyze for Text (OCR) =====
result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.READ],
    language="en"
)

# ===== Process and Print Extracted Text =====
if result.read is not None:
    print("📄 Extracted Text:\n")
    for block in result.read.blocks:
        for line in block.lines:
            print(f"Line: {line.text}")
            for word in line.words:
                print(f"  → Word: {word.text} (Confidence: {word.confidence * 100:.2f}%)")
else:
    print("No text found in the image.")

# ===== Optional: Draw Bounding Boxes on Image =====
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

if result.read is not None:
    for block in result.read.blocks:
        for line in block.lines:
            if hasattr(line, "bounding_polygon"):
                poly = line.bounding_polygon
                points = [(p.x, p.y) for p in poly]
                draw.polygon(points, outline="green", width=3)

# Save annotated image
output_path = r"C:\Users\Nizam\Desktop\azure_vision\text_detected_output.jpg"
image.save(output_path)
print(f"\n✅ Annotated image saved to: {output_path}")
image.show()
