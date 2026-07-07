Company: CODTECH IT SOLUTIONS PVT. LTD
Name: Moode Hemanth Naik
Intern ID: CITS1163
Domain: Artificial Intelligence
Duration: 8 Weeks
Internship Period: 17 May 2026 - 12 July 2026
Mentor: Neela Santhosh Kumar

📝 Task 1: Text Summarization Tool
This repository features an automated Text Summarization Tool built using modern Natural Language Processing (NLP) techniques. It leverages the state-of-the-art BART (Bidirectional and Auto-Regressive Transformers) architecture via the Hugging Face transformers library to compress lengthy text articles into concise, coherent summaries while preserving core semantic meaning.
📋 Project Overview
Manual summaries of lengthy research papers, financial reports, or news articles are time-consuming. This tool automates the distillation process using a pre-trained sequence-to-sequence model (facebook/bart-large-cnn), which is fine-tuned specifically for text summarization tasks.
Core Pipeline
Model Loading: Dynamically initializes the pre-trained tokenizers and model weights.
Text Chunking / Truncation: Gracefully handles long text constraints using dynamic parameter inputs.
Abstractive Inference: The model generates brand-new sentence structures to summarize the text natively rather than just pulling raw sentences out (extractive approach).
Execution Comparison: Cleans and outputs the original vs. summary text for analytical review.
🚀Setup & Installation
1. System Requirements
Python 3.8 or higher
Recommended: PyTorch (CPU version is sufficient for running small test iterations, but a CUDA-enabled GPU dramatically speeds up generation).
2. Dependency Installation
Create a virtual environment and install the required modules.
Bash
# Clone or create your directory
mkdir text-summarizer && cd text-summarizer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install transformers torch sentencepiece
Alternatively, you can save this as requirements.txt:
Plaintext
transformers>=4.30.0
torch>=2.0.0
sentencepiece>=0.1.99
And run: pip install -r requirements.txt
💻 Implementation Source Code
Save the following code block as text_summarizer.py. It is built with safety thresholds, explicit type hinting, and clear text blocks.
Python
"""
Text Summarization Tool using Hugging Face Transformers.
Author: [Your Name / Intern ID]
Domain: Machine Learning / NLP
Model Architecture: BART (facebook/bart-large-cnn)
"""

import sys
import os
from transformers import BartTokenizer, BartForConditionalGeneration

# Suppress unnecessary framework warning cleanups
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class TextSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initializes the BART Tokenizer and Model for conditional generation.
        """
        print(f"[INFO] Initializing Transformer pipeline using: {model_name}...")
        try:
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            print("[INFO] Model and weights successfully cached and loaded.\n")
        except Exception as e:
            print(f"[ERROR] Failed to load model architecture: {str(e)}")
            sys.exit(1)

    def generate_summary(self, text: str, max_len: int = 150, min_len: int = 40) -> str:
        """
        Tokenizes inputs, generates abstractive summary IDs, and returns decoded string text.
        """
        if not text.strip():
            return "Error: Input text string is blank."

        # Tokenize and encode input document
        # 'pt' returns PyTorch tensors
        inputs = self.tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")

        # Generate Summary IDs using Beam Search optimization parameters
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_len,
            min_length=min_len,
            early_stopping=True
        )

        # Decode summary tokens back to human-readable string text
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def main():
    # Sample lengthy article text regarding artificial intelligence advancements
    sample_article = (
        "Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, "
        "particularly computer systems. It is a field of research in computer science that develops "
        "and studies methods and software which enable machines to perceive their environment and use "
        "learning and intelligence to take actions that maximize their chances of achieving defined goals. "
        "Such machines may be called AIs. AI technology is widely used throughout industry, government, "
        "and science. Some high-profile applications include advanced web search engines (e.g., Google Search); "
        "recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech "
        "(such as Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative "
        "and creative tools (ChatGPT and AI art); and automated decision-making. "
        "Many AI applications are not perceived as AI: 'A lot of cutting-edge AI has filtered into "
        "general applications, often without being called AI because once something becomes useful "
        "enough and common enough it's not labeled AI anymore,' known as the AI effect. "
        "Alan Turing was the first person to conduct substantial research in the field that he called "
        "machine intelligence. AI research was founded as an academic discipline in 1956. The field "
        "went through multiple cycles of optimism, followed by disappointment and loss of funding (known "
        "as an 'AI winter'), followed by new approaches, success, and renewed funding."
    )

    # Initialize the engine
    summarizer = TextSummarizer()

    # Process and track inferences
    print("=" * 40 + " ORIGINAL ARTICLE " + "=" * 40)
    print(sample_article.strip())
    print("=" * 98 + "\n")

    print("[INFO] Processing abstractive summarization. Please wait...")
    summary_result = summarizer.generate_summary(sample_article, max_len=120, min_len=30)

    print("=" * 40 + " GENERATED SUMMARY " + "=" * 40)
    print(summary_result)
    print("=" * 98)


if __name__ == "__main__":
    main()

    📈 Example Execution Output
When you execute the script running python text_summarizer.py, your terminal console will print out the original input array compared to the concise, generated sentence weights:
Plaintext
[INFO] Initializing Transformer pipeline using: facebook/bart-large-cnn...
[INFO] Model and weights successfully cached and loaded.

======================================== ORIGINAL ARTICLE ========================================
Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particularly computer systems. It is a field of research in computer science that develops and studies methods and software which enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. Such machines may be called AIs. AI technology is widely used throughout industry, government, and science. Some high-profile applications include advanced web search engines (e.g., Google Search); recommendation systems (used by YouTube, Amazon, and Netflix); interacting via human speech (such as Google Assistant, Siri, and Alexa); autonomous vehicles (e.g., Waymo); generative and creative tools (ChatGPT and AI art); and automated decision-making. Many AI applications are not perceived as AI: 'A lot of cutting-edge AI has filtered into general applications, often without being called AI because once something becomes useful enough and common enough it's not labeled AI anymore,' known as the AI effect. Alan Turing was the first person to conduct substantial research in the field that he called machine intelligence. AI research was founded as an academic discipline in 1956. The field went through multiple cycles of optimism, followed by disappointment and loss of funding (known as an 'AI winter'), followed by new approaches, success, and renewed funding.
==================================================================================================

[INFO] Processing abstractive summarization. Please wait...
======================================== GENERATED SUMMARY ========================================
Artificial intelligence (AI) is intelligence exhibited by machines, particularly computer systems. AI technology is widely used throughout industry, government, and science. High-profile applications include advanced web search engines, recommendation systems, and autonomous vehicles. AI research was founded as an academic discipline in 1956.
==================================================================================================
============================================XXXXXXXXXX============================================
