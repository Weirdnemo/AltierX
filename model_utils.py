from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class ResearchAssistant:
    def __init__(self):
        # Initialize the model and tokenizer
        self.model_name = "facebook/opt-1.3b"  # Using a larger model for better quality
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def generate_outline(self, topic):
        prompt = f"""Create a detailed research paper outline for the topic: {topic}

The outline should follow this structure:
1. Introduction
   - Background
   - Problem Statement
   - Research Objectives
2. Literature Review
   - Previous Work
   - Current State of Research
3. Methodology
   - Data Collection
   - Machine Learning Approach
   - Evaluation Metrics
4. Results and Discussion
   - Analysis
   - Findings
5. Conclusion
   - Summary
   - Future Work

Outline:"""
        response = self.generator(
            prompt,
            max_length=1000,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        return response[0]['generated_text']

    def generate_abstract(self, topic, key_points):
        prompt = f"""Write a professional academic abstract for a research paper on {topic}.

Key points to include:
{key_points}

The abstract should be concise, clear, and follow the standard academic format including:
- Problem statement
- Methodology
- Key findings
- Implications

Abstract:"""
        response = self.generator(
            prompt,
            max_length=300,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        return response[0]['generated_text']

    def generate_literature_review(self, topic, papers):
        prompt = f"""Write a comprehensive literature review for {topic} based on these papers:

{papers}

The literature review should:
1. Synthesize the key findings
2. Identify research gaps
3. Compare different approaches
4. Discuss implications

Literature Review:"""
        response = self.generator(
            prompt,
            max_length=1000,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        return response[0]['generated_text']

    def generate_key_points(self, text):
        prompt = f"""Extract and organize the key points from the following text:

{text}

Format the key points as a structured list with:
- Main points
- Supporting evidence
- Implications

Key Points:"""
        response = self.generator(
            prompt,
            max_length=300,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        return response[0]['generated_text'] 