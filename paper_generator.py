from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class PaperGenerator:
    def __init__(self, model_name="facebook/opt-1.3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def generate_outline(self, topic, keywords=None):
        prompt = f"""Create a detailed research paper outline for the topic: {topic}.
Keywords: {keywords if keywords else ''}

Outline:"""
        response = self.generator(prompt, max_length=400, temperature=0.7, do_sample=True, top_p=0.9, repetition_penalty=1.2)
        return response[0]['generated_text']

    def generate_section(self, section_name, topic, title, keywords=None, instructions=None):
        prompt = f"""Write the {section_name} section for a research paper.
Title: {title}
Topic: {topic}
Keywords: {keywords if keywords else ''}
Instructions: {instructions if instructions else ''}

{section_name}:"""
        response = self.generator(prompt, max_length=600, temperature=0.7, do_sample=True, top_p=0.9, repetition_penalty=1.2)
        return response[0]['generated_text']

    def generate_full_paper(self, topic, title, keywords=None, instructions=None):
        sections = ["Introduction", "Literature Review", "Methodology", "Results and Discussion", "Conclusion"]
        paper = ""
        for section in sections:
            paper += f"\n\n## {section}\n"
            paper += self.generate_section(section, topic, title, keywords, instructions)
        return paper 