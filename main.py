import streamlit as st
from datetime import date
from paper_generator import PaperGenerator

st.set_page_config(page_title="AI Agent for Academic Research", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Your AI Academic Research Assistant")
    st.write("AI-Powered Paper Generation")
    st.text_input("Search for academic papers, journals, or information...")
    st.button("Search")
    st.markdown("**About**")
    st.info("This tool generates a personalized academic research paper based on your inputs. Fill in the form and let our specialized agents craft your paper!")
    st.markdown("**How it works**")
    st.markdown("""
    1. Enter your research details  
    2. AI conducts literature research  
    3. Generate a paper outline  
    4. Draft and edit your paper  
    5. Download your final paper  
    """)

# Main
st.title("Your AI Agent for Academic Research")
st.subheader("Generate your personalized research paper with AI-powered academic agents.")

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
    include_refs = st.checkbox("Include web search for latest references", value=True)
    submitted = st.form_submit_button("Generate My Research Paper")

if submitted:
    st.info("Generating your research paper... (this may take a while)")
    generator = PaperGenerator()
    outline = generator.generate_outline(topic, keywords)
    st.markdown("#### Paper Outline")
    st.markdown(outline)
    paper = generator.generate_full_paper(topic, title, keywords, instructions)
    st.success("Your research paper is ready!")
    st.markdown("#### Generated Paper")
    st.text_area("Full Paper", paper, height=400)
    st.download_button("Download Paper", paper, file_name="research_paper.txt") 