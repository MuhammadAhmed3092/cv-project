import streamlit as st
import json
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_model.eval()

# Initialize ResNet model for image feature extraction
resnet_model = models.resnet101(pretrained=True)
resnet_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess image
def preprocess_image(image):
    """Preprocess the input image."""
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to extract features (as defined earlier)
def extract_features(image):
    """Extract features from the image using ResNet."""
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        features = resnet_model(image_tensor)
    return features


# Function to generate a report using BERT
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_report(image):
    """Generate a report based on image features and closest matching annotation."""
    features = extract_features(image)  # Extract features from the image
    
    # Preprocess annotation reports
    annotation_embeddings = []
    annotation_texts = []

    # Loop through the dataset and get reports from the list of dictionaries
    for annotation in annotations["train"]:
        report = annotation["report"]  # Get the report text
        report_id = annotation["id"]  # Get the report id (optional, can be used for logging)
        
        # Tokenize and embed annotation reports
        inputs = tokenizer(report, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            # Example to obtain the report embeddings (can be changed based on your model)
            embedding = resnet_model.fc(inputs['input_ids']).mean(dim=1)  # Adjust this to your actual model
        annotation_embeddings.append(embedding.numpy())
        annotation_texts.append(report)

    # Convert annotation embeddings to a matrix
    annotation_matrix = np.vstack(annotation_embeddings)

    # Compute similarity between image features and annotations
    with torch.no_grad():
        image_embedding = features.mean(dim=1).numpy()  # Use the extracted features
    similarities = cosine_similarity(image_embedding, annotation_matrix)

    # Find the most similar report
    most_similar_idx = np.argmax(similarities)
    return annotation_texts[most_similar_idx]  # Return the closest report
# Function to normalize text
def normalize_text(text):
    return text.lower().strip()

# Function to compute BLEU score using corpus_bleu
def compute_bleu(reference_reports, generated_reports):
    """
    Compute the BLEU score for generated reports against reference reports.
    """
    reference_reports = [[normalize_text(ref).split()] for ref in reference_reports]  # Tokenize references
    generated_reports = [normalize_text(gen).split() for gen in generated_reports]  # Tokenize predictions
    smoothie = SmoothingFunction().method4
    return corpus_bleu(reference_reports, generated_reports, smoothing_function=smoothie)

# Load annotations from JSON
@st.cache_data
def load_annotations(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

# Example annotation file path (replace with your path)
annotations = load_annotations("annotation.json")  # Path to your annotation.json

reference_reports = []
for dataset in ["train", "test", "val"]:
    if dataset in annotations:
        # Directly extend if it's a list
        reference_reports.extend(annotations[dataset])


# Streamlit app layout
st.markdown("<h1 style='color: #2E86C1; text-align: center;'>Chest X-Ray Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a chest X-ray image to generate a report and evaluate its quality with a BLEU score.</p>", unsafe_allow_html=True)

# Image upload
uploaded_image = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    try:
        # Load and display the image
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate report
        st.markdown("<h3 style='color: #117A65;'>Generated Report</h3>", unsafe_allow_html=True)
        generated_report = generate_report(image)
        st.markdown(f"<div style='background-color: #2e86c1; padding: 10px; border-radius: 5px;'>{generated_report}</div>", unsafe_allow_html=True)
        
        # BLEU score calculation
        st.markdown("<h3 style='color: #B9770E;'>BLEU Score</h3>", unsafe_allow_html=True)
        
        # Compute BLEU score
        bleu_score = compute_bleu(reference_reports, [generated_report])
        st.markdown(f"<div style='background-color: #26da6c; padding: 10px; border-radius: 5px; text-align: center;'><b>{bleu_score:.4f}</b></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
