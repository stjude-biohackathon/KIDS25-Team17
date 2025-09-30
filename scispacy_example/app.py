
import spacy_streamlit
import spacy
import streamlit as st
from typing import Sequence
from utils import fetch_pubmed_text

# Enable GPU support for spaCy
spacy.prefer_gpu()

email = ""

st.set_page_config(layout="wide")

#text input for pmid
pmid = st.text_input("Enter PubMed ID (PMID)", "")

#default text and clean-up of default text
DEFAULT_TEXT = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."""

spacy_model = "en_ner_bionlp13cg_md"

st.title("Publication Entity Annotation")
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)

if pmid:
    pmid = pmid.strip()
    if not pmid.isdigit():
        st.error("Please enter a valid PubMed ID (PMID) consisting of digits only.")
        st.stop()
    pubmed_text = fetch_pubmed_text(pmid, email)
    if pubmed_text:
        text = pubmed_text

#test the pmid code
clean_text = text.replace('\n', '')
doc = spacy_streamlit.process_text(spacy_model, clean_text)

#define the labels detected in the text
labels: Sequence[str] = tuple()
labels = labels or list({ent.label_ for ent in doc.ents})

# Create distict colours for labels
col_dict = {}
s_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
for label, colour in zip(labels, s_colours):
    col_dict[label] = colour

#visualize the NER data
spacy_streamlit.visualize_ner(
    doc,
    # labels=["PERSON", "DATE", "GPE"],
    show_table=False,
    title="Select entities to visualize",
    colors=col_dict,
)
st.text(f"Analyzed using spaCy model {spacy_model}")