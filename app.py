import streamlit as st
from model_utils import ResearchAssistant
import time
import torch

# Set page config
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize the research assistant
@st.cache_resource
def load_model():
    return ResearchAssistant()

# App title and description
st.title("📚 Research Paper Assistant")
st.markdown("""
This tool helps you write research papers using AI assistance. Choose from the following options:
- Generate research paper outline
- Create an abstract
- Write a literature review
- Extract key points from text
""")

# Initialize session state for storing generated content
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = ""

# Sidebar for model information
with st.sidebar:
    st.header("About")
    st.info("""
    This application uses Hugging Face's language models to assist in research paper writing.
    The generated content should be reviewed and edited as needed.
    """)
    
    st.header("Model Information")
    st.write("Model: facebook/opt-1.3b")
    st.write("Device: " + ("GPU" if torch.cuda.is_available() else "CPU"))

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Outline Generator",
    "📝 Abstract Generator",
    "📚 Literature Review",
    "🔑 Key Points Extractor"
])

with tab1:
    st.header("Research Paper Outline Generator")
    topic = st.text_input("Enter your research topic:", key="outline_topic")
    if st.button("Generate Outline", key="outline_btn"):
        if topic:
            with st.spinner("Generating outline..."):
                assistant = load_model()
                outline = assistant.generate_outline(topic)
                st.session_state.generated_content = outline
                st.text_area("Generated Outline:", outline, height=400)
        else:
            st.warning("Please enter a research topic.")

with tab2:
    st.header("Abstract Generator")
    topic = st.text_input("Enter your research topic:", key="abstract_topic")
    key_points = st.text_area("Enter key points (one per line):", key="abstract_points")
    if st.button("Generate Abstract", key="abstract_btn"):
        if topic and key_points:
            with st.spinner("Generating abstract..."):
                assistant = load_model()
                abstract = assistant.generate_abstract(topic, key_points)
                st.session_state.generated_content = abstract
                st.text_area("Generated Abstract:", abstract, height=400)
        else:
            st.warning("Please enter both topic and key points.")

with tab3:
    st.header("Literature Review Generator")
    topic = st.text_input("Enter your research topic:", key="lit_topic")
    papers = st.text_area("Enter paper summaries (one per line):", key="lit_papers")
    if st.button("Generate Literature Review", key="lit_btn"):
        if topic and papers:
            with st.spinner("Generating literature review..."):
                assistant = load_model()
                review = assistant.generate_literature_review(topic, papers)
                st.session_state.generated_content = review
                st.text_area("Generated Literature Review:", review, height=400)
        else:
            st.warning("Please enter both topic and paper summaries.")

with tab4:
    st.header("Key Points Extractor")
    text = st.text_area("Enter your text:", key="key_points_text")
    if st.button("Extract Key Points", key="key_points_btn"):
        if text:
            with st.spinner("Extracting key points..."):
                assistant = load_model()
                points = assistant.generate_key_points(text)
                st.session_state.generated_content = points
                st.text_area("Extracted Key Points:", points, height=400)
        else:
            st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Hugging Face") 