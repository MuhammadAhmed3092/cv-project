import torch
from torchvision import models, transforms
from PIL import Image
from transformers import AutoTokenizer
import json
from nltk.translate.bleu_score import corpus_bleu


class ReportGenerator:
    def __init__(self, tokenizer_model="bert-base-uncased"):
        """
        Initialize the report generator with a ResNet model and tokenizer.
        Args:
            tokenizer_model (str): Hugging Face tokenizer model name.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        
        # Load ResNet101 pre-trained model
        self.resnet_model = models.resnet101(pretrained=True).to(self.device)
        self.resnet_model.eval()
        
        # Define image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path):
        """
        Preprocess and prepare an image for feature extraction.
        Args:
            image_path (str): Path to the input image.
        Returns:
            torch.Tensor: Transformed image tensor.
        """
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def extract_features(self, image_path):
        """
        Extract features from an image using ResNet.
        Args:
            image_path (str): Path to the input image.
        Returns:
            torch.Tensor: Extracted features.
        """
        img_tensor = self.process_image(image_path)
        with torch.no_grad():
            features = self.resnet_model(img_tensor)
        return features

    def tokenize_reports(self, reports):
        """
        Tokenize a list of reports using the tokenizer.
        Args:
            reports (list of str): List of textual reports.
        Returns:
            list of torch.Tensor: Tokenized report tensors.
        """
        tokenized_reports = []
        for report in reports:
            encoded = self.tokenizer(report, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            tokenized_reports.append(encoded)
        return tokenized_reports

    def compute_bleu(self, generated_reports, reference_reports):
        """
        Compute BLEU score for generated reports against reference reports.
        Args:
            generated_reports (list of str): Generated reports.
            reference_reports (list of list of str): Reference reports.
        Returns:
            float: BLEU score.
        """
        return corpus_bleu(reference_reports, generated_reports)


def load_dataset(dataset_path):
    """
    Load the dataset from a JSON file.
    Args:
        dataset_path (str): Path to the dataset file.
    Returns:
        dict: Parsed dataset.
    """
    with open("C:\\Users\muham\Downloads\x-rgen\data\annotation_example.json", "r") as f:
        return json.load(f)
