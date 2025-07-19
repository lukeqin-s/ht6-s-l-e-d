print("Starting program...")

print("Loading OpenCV...")
import cv2
print("✓ OpenCV loaded")

print("Loading other packages...")
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
print("✓ All packages loaded")

# Load API key from .env
print("Loading environment variables...")
load_dotenv()
api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY not found in .env file")
print("✓ API key loaded")

# Capture image from webcam
print("Initializing camera...")
cap = cv2.VideoCapture(0)
print("✓ Camera initialized")

print("Capturing image...")
ret, frame = cap.read()
print("✓ Image captured")

print("Releasing camera...")
cap.release()
print("✓ Camera released")

if not ret:
    raise Exception("Failed to capture image from camera.")

# Save the image
print("Saving image...")
image_path = "captured.jpg"
cv2.imwrite(image_path, frame)
print(f"✓ Image saved as {image_path}")

# Run inference
print("Setting up AI client...")
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)
print("✓ Client initialized")

print("Sending image to AI model...")
result = CLIENT.infer(image_path, model_id="direction-nsuqs/1")
print("✓ AI processing complete")

print("Results:")
print(result)