import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class LocalLLMBackend:
    """
    AltierX Local Backend: Uses a local instruction-tuned LLM (Mistral-7B-Instruct or Falcon-7B-Instruct) for high-quality academic text generation.
    Requires sufficient RAM/VRAM (at least 16GB RAM, 8GB+ VRAM recommended for GPU).
    """
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}'. Ensure you have enough RAM/VRAM and the model is downloaded. Error: {e}")

    def generate_outline(self, topic, keywords=None):
        prompt = f"""Create a detailed research paper outline for the topic: {topic}.
Keywords: {keywords if keywords else ''}

Outline:"""
        return self._generate(prompt, max_length=256)

    def generate_section(self, section_name, topic, title, keywords=None, instructions=None):
        prompt = f"""Write the {section_name} section for a research paper.
Title: {title}
Topic: {topic}
Keywords: {keywords if keywords else ''}
Instructions: {instructions if instructions else ''}

{section_name}:"""
        return self._generate(prompt, max_length=512)

    def generate_full_paper(self, topic, title, keywords=None, instructions=None):
        sections = ["Introduction", "Literature Review", "Methodology", "Results and Discussion", "Conclusion"]
        paper = ""
        for section in sections:
            paper += f"\n\n## {section}\n"
            paper += self.generate_section(section, topic, title, keywords, instructions)
        return paper

    def _generate(self, prompt, max_length=512):
        try:
            response = self.generator(
                prompt,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                num_return_sequences=1
            )
            return response[0]["generated_text"]
        except Exception as e:
            return f"[Error generating text: {e}]" 