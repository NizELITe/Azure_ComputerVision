import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from PIL import Image, ImageDraw

# ==== Azure Vision Config ====
endpoint = ""
key = ""

# Create the Image Analysis client
cv_client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

# ==== Image to Analyze ====
image_file = r"C:\Users\Nizam\Desktop\azure_vision\imageforcaption.jpeg"

# Read image bytes
with open(image_file, "rb") as f:
    image_data = f.read()

print(f"\nAnalyzing {image_file}\n")

# ==== Analyze Image ====
result = cv_client.analyze(
    image_data=image_data,
    visual_features=[
        VisualFeatures.OBJECTS,
        VisualFeatures.PEOPLE,
    ],
)

# ==== Annotate and Print Results ====
image = Image.open(image_file)
draw = ImageDraw.Draw(image)

# --- Objects ---
if result.objects is not None:
    print("\nObjects detected:")
    for obj in result.objects.list:
        tag = obj.tags[0].name
        confidence = obj.tags[0].confidence * 100
        box = obj.bounding_box
        print(f" - {tag} (confidence: {confidence:.2f}%)")
        draw.rectangle(
            [(box.x, box.y), (box.x + box.width, box.y + box.height)],
            outline="red",
            width=3
        )
        draw.text((box.x, box.y - 10), tag, fill="red")

# --- People ---
if result.people is not None:
    print("\nPeople detected:")
    for person in result.people.list:
        confidence = person.confidence * 100
        box = person.bounding_box
        print(f" - Person (confidence: {confidence:.2f}%)")
        draw.rectangle(
            [(box.x, box.y), (box.x + box.width, box.y + box.height)],
            outline="blue",
            width=3
        )

# ==== Save and Show Output ====
output_path = r"C:\Users\Nizam\Desktop\azure_vision\annotated_output.jpg"
image.save(output_path)
print(f"\n✅ Annotated image saved as: {output_path}")

# Optional: Show the image
image.show()
