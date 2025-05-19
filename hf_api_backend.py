import os
import requests

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

class HFAPIBackend:
    """AltierX backend: Uses Hugging Face Inference API for fast, high-quality academic text generation."""
    def __init__(self, model_url=API_URL):
        self.api_url = model_url
        self.headers = HEADERS

    def query(self, prompt, max_length=512):
        """Query the Hugging Face Inference API for text generation (AltierX)."""
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_length},
            "options": {"wait_for_model": True}
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result[0]["summary_text"] if isinstance(result, list) and "summary_text" in result[0] else result[0].get("generated_text", "")

    def generate_outline(self, topic, keywords=None):
        """Generate a research paper outline using AltierX."""
        prompt = f"""Create a detailed research paper outline for the topic: {topic}.
Keywords: {keywords if keywords else ''}

Outline:"""
        return self.query(prompt, max_length=256)

    def generate_section(self, section_name, topic, title, keywords=None, instructions=None):
        """Generate a specific section of a research paper using AltierX."""
        prompt = f"""Write the {section_name} section for a research paper.
Title: {title}
Topic: {topic}
Keywords: {keywords if keywords else ''}
Instructions: {instructions if instructions else ''}

{section_name}:
"""
        return self.query(prompt, max_length=512)

    def generate_full_paper(self, topic, title, keywords=None, instructions=None):
        """Generate a full research paper using AltierX."""
        sections = ["Introduction", "Literature Review", "Methodology", "Results and Discussion", "Conclusion"]
        paper = ""
        for section in sections:
            paper += f"\n\n## {section}\n"
            paper += self.generate_section(section, topic, title, keywords, instructions)
        return paper 