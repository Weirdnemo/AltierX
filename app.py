import streamlit as st
from model_utils import ResearchAssistant
import time
import torch
import platform
from datetime import date
from local_backend import LocalLLMBackend

# Disable hot reload warning
st.set_page_config(page_title="AltierX - AI Academic Research Assistant", layout="wide", initial_sidebar_state="expanded")

# Hero Section
st.markdown(
    """
    <div style='background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2em 1em; border-radius: 12px; margin-bottom: 2em; color: #fff; text-align: center;'>
        <h1 style='margin-bottom: 0.2em;'>AltierX</h1>
        <h3 style='margin-top: 0;'>Your Offline AI Agent for Academic Research</h3>
        <p>Generate outlines, sections, and full research papers with state-of-the-art local LLMs. <b>No cloud. No data leaves your machine.</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize the research assistant
@st.cache_resource
def load_model():
    return ResearchAssistant()

# Sidebar: System Status
with st.sidebar:
    st.header("System Status")
    st.write(f"**Model:** mistralai/Mistral-7B-Instruct-v0.2")
    st.write(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.write(f"**Python:** {platform.python_version()}")
    st.write(f"**PyTorch:** {torch.__version__}")
    st.write(f"**RAM:** {round((psutil.virtual_memory().total/1e9), 1)} GB")
    st.write(f"**VRAM:** {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "N/A")
    st.markdown("---")
    st.info("Hot reload is disabled for stability. Please manually refresh the page after code changes.")

# Tabs for workflow
tab1, tab2, tab3 = st.tabs(["📝 Generate Paper", "ℹ️ How it Works", "📖 About AltierX"])

with tab1:
    with st.form("paper_form"):
        st.subheader("Generate Your Research Paper")
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Research Topic", placeholder="e.g., Deep Learning in Healthcare")
            title = st.text_input("Paper Title", placeholder="e.g., Advances in Deep Learning for Medical Diagnosis")
            due_date = st.date_input("Due Date", value=date.today())
            instructions = st.text_area("Additional Instructions", placeholder="Any additional instructions or requirements…")
        with col2:
            length = st.slider("Paper Length (pages)", 5, 50, 10)
            paper_type = st.selectbox("Paper Type", ["Journal", "Conference", "Thesis"])
            style = st.selectbox("Writing Style", ["Formal", "Informal"])
            keywords = st.text_input("Keywords/Focus (comma separated)")
        submitted = st.form_submit_button("🚀 Generate My Research Paper")

    if submitted:
        st.info("Generating your research paper... This may take a while, especially on CPU.")
        generator = LocalLLMBackend()
        with st.spinner("Generating outline..."):
            outline = generator.generate_outline(topic, keywords)
        st.success("Outline generated!")
        st.markdown(f"<div style='background:#222;padding:1em;border-radius:8px;color:#fff'>{outline}</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### Generated Paper (Section by Section)")
        sections = ["Introduction", "Literature Review", "Methodology", "Results and Discussion", "Conclusion"]
        paper = ""
        for section in sections:
            with st.spinner(f"Generating {section}..."):
                section_text = generator.generate_section(section, topic, title, keywords, instructions)
                st.markdown(f"**{section}:**")
                st.markdown(f"<div style='background:#f8f9fa;padding:1em;border-radius:8px'>{section_text}</div>", unsafe_allow_html=True)
                paper += f"\n\n## {section}\n{section_text}"
        st.success("Your research paper is ready!")
        st.download_button("Download Paper", paper, file_name="research_paper.txt")

with tab2:
    st.markdown("""
    1. Enter your research details in the form.
    2. AltierX generates an outline and each section using a local LLM.
    3. Download your full paper as a text file.
    4. All processing is offline—your data never leaves your machine.
    """)

with tab3:
    st.markdown("""
    **AltierX** is a fully offline, open-source academic research assistant powered by state-of-the-art local language models.  
    - No cloud. No API keys.  
    - All computation is local and private.
    - Built with ❤️ using Streamlit and Hugging Face Transformers.
    """)

# Show a warning if hot reload is off
st.warning("Hot reload is disabled for stability. Please manually refresh the page after code changes.", icon="⚠️") 