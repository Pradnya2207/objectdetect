from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import gradio as gr

# Load the model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Define the function for prediction
def recognize_image(image):
    # Convert the input image to RGB
    image = Image.fromarray(image).convert("RGB")
    
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Make predictions
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    
    # Get the predicted class label
    return model.config.id2label[predicted_class_idx]

# Create a Gradio interface
app = gr.Interface(
    fn=recognize_image,                 # Prediction function
    inputs=gr.Image(type="numpy"),      # Input: Image
    outputs="text",                     # Output: Predicted label
    title="Image Recognition App"       # App title
)

# Launch the app
app.launch()