from transformers import pipeline
from PIL import Image

# load model from Hugging Face
classifier = pipeline(
    "image-classification",
    model="dima806/yoga_pose_image_classification"
)

# load your image
image = Image.open("beer_test.jpg")

# predict pose
result = classifier(image)

print("Prediction:")
print(result)