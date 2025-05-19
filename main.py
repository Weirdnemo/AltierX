import streamlit as st
from datetime import date
from local_backend import LocalLLMBackend

st.set_page_config(page_title="AltierX - AI Academic Research Assistant", layout="wide", initial_sidebar_state="expanded")

# Sidebar
with st.sidebar:
    st.header("AltierX: Your AI Academic Research Assistant 🧑‍💻")
    st.markdown("""
    <style>
    .sidebar-content {font-size: 1.1em;}
    </style>
    """, unsafe_allow_html=True)
    st.write("AI-Powered Paper Generation")
    st.text_input("Search for academic papers, journals, or information...")
    st.button("Search")
    st.markdown("**About**")
    st.info("AltierX generates personalized academic research papers based on your inputs. Fill in the form and let our specialized agents craft your paper!")
    st.markdown("**How it works**")
    st.markdown("""
    1. Enter your research details  
    2. AI conducts literature research  
    3. Generate a paper outline  
    4. Draft and edit your paper  
    5. Download your final paper  
    """)
    st.markdown("---")
    st.markdown("**Model:** mistralai/Mistral-7B-Instruct-v0.2 (local)")
    st.markdown("**Device:** " + ("GPU" if st.runtime.exists() and st.runtime.get_instance()._is_running_with_streamlit and hasattr(st, 'cuda') and st.cuda.is_available() else "CPU"))
    st.markdown("---")
    st.caption("AltierX is fully offline. All data stays on your machine.")

# Main
st.title("AltierX: AI Agent for Academic Research 🧑‍💻")
st.subheader("Generate your personalized research paper with AltierX's AI-powered academic agents.")

with st.form("paper_form"):
    st.write("### Generate Your Research Paper")
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
    include_refs = st.checkbox("Include web search for latest references (not available in offline mode)", value=False, disabled=True)
    submitted = st.form_submit_button("Generate My Research Paper")

if submitted:
    st.info("Generating your research paper... This may take a while, especially on CPU.")
    generator = LocalLLMBackend()
    with st.spinner("Generating outline..."):
        outline = generator.generate_outline(topic, keywords)
    st.markdown("#### Paper Outline")
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
    st.markdown("---")
    st.markdown("#### Download Full Paper")
    st.download_button("Download Paper", paper, file_name="research_paper.txt") 