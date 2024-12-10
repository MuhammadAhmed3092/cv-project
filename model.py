from transformers import AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import torch
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Initialize the tokenizer and ResNet model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
resnet_model = models.resnet101(pretrained=True)
resnet_model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the reports dataset
csv_path = "indiana_reports.csv"
reports_df = pd.read_csv(csv_path)

# Process the image and extract features
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Simulate generating diverse findings based on categories
def generate_findings():
    categories = ['Cardiac', 'Lung', 'Bone', 'Abdomen', 'Soft Tissue', 'Vascular']
    
    # Randomly select category and generate findings
    category = random.choice(categories)
    if category == 'Cardiac':
        findings = "Normal cardiac silhouette, no signs of heart failure."
    elif category == 'Lung':
        findings = "Mild bilateral lung opacities, likely due to pneumonia."
    elif category == 'Bone':
        findings = "No fractures or dislocations detected in the bones."
    elif category == 'Abdomen':
        findings = "No abnormalities in the abdominal organs."
    elif category == 'Soft Tissue':
        findings = "No signs of soft tissue swelling or masses."
    elif category == 'Vascular':
        findings = "Normal vascular structure, no signs of aneurysm."
    
    return findings, category

# Generate the BLEU score based on the generated report and reference reports
def generate_bleu_score(generated_report, reference_reports):
    # Tokenize the sentences
    reference_tokens = [report.split() for report in reference_reports]
    generated_tokens = generated_report.split()
    
    # Calculate BLEU score using sentence_bleu
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=SmoothingFunction().method7)
    
    return round(bleu_score, 3)

# Generate BLEU score and report
def generate_bleu_score_and_report(image_path, reference_reports):
    # Process image
    image = process_image(image_path)
    
    # Extract ResNet features (if needed, this can be extended to generate features)
    with torch.no_grad():
        features = resnet_model(image)
    
    # Generate findings based on category
    findings, category = generate_findings()
    
    # Clean findings by replacing 'XXXX' with 'unknown' (if applicable)
    cleaned_findings = findings.replace("XXXX", "")
    
    # Generate the report based on findings
    generated_report = f"Category: {category}\nFindings: {cleaned_findings}"
    
    # Generate BLEU score based on the comparison with reference reports
    bleu_score = generate_bleu_score(generated_report, reference_reports)
    
    # Create a detailed report
    report = f"""
    Report for Uploaded X-ray Image:
    ---------------------------------
    Category: {category}
    Findings: {cleaned_findings}
    BLEU Score: {bleu_score}
    """
    
    return bleu_score, report
