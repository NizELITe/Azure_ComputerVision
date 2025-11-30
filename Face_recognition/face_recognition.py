from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# ===== Load environment variables =====
load_dotenv(dotenv_path=r"C:\Users\Nizam\Desktop\azure_vision\.env")
endpoint = os.getenv("AZURE_ENDPOINTFace")
key = os.getenv("AZURE_KEYFace")

# ===== Initialize Face Client =====
face_client = FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# ===== Facial Attributes =====
features = [
    FaceAttributeTypeDetection01.HEAD_POSE,
    FaceAttributeTypeDetection01.OCCLUSION,
    FaceAttributeTypeDetection01.ACCESSORIES
]

# ===== Detect Faces =====
with open(r"C:\Users\Nizam\Desktop\azure_vision\group_pic.jpg", "rb") as image_data:
    detected_faces = face_client.detect(
        image_content=image_data.read(),
        detection_model=FaceDetectionModel.DETECTION01,
        recognition_model=FaceRecognitionModel.RECOGNITION01,
        return_face_id=True,   # <-- Required argument added
        return_face_attributes=features
    )

# ===== Print Results =====
if not detected_faces:
    print("⚠️ No faces detected.")
else:
    for i, face in enumerate(detected_faces):
        print(f"👤 Face {i+1}:")
        print(f"  Bounding box: {face.face_rectangle}")
        print(f"  Head pose: {face.face_attributes.head_pose}")
        print(f"  Occlusion: {face.face_attributes.occlusion}")
        print(f"  Accessories: {face.face_attributes.accessories}")
