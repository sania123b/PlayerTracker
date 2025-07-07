import cv2
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Test image
img = cv2.imread("test_image.jpg")  # Save a frame from your video as test_image.jpg
if img is None:
    print("ERROR: Could not load test image!")
    exit()

# Run detection
results = model(img, conf=0.3, classes=[0])  # Low confidence threshold

# Visualize
if len(results[0].boxes) == 0:
    print("NO DETECTIONS! Possible issues:")
    print("1. Wrong model (not trained for players)")
    print("2. Players are too small in frame")
    print("3. Lighting/quality issues")
else:
    annotated = results[0].plot()  # Draw detections
    cv2.imwrite("detection_test.jpg", annotated)
    print(f"Found {len(results[0].boxes)} players - see detection_test.jpg")