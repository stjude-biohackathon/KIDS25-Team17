# conda env:  /mnt/scratch2/Maggie/miniconda3/envs/scispacy
import spacy_streamlit
import spacy
import streamlit as st
from typing import Sequence
import pandas as pd

# Enable GPU support for spaCy
spacy.prefer_gpu()

st.set_page_config(layout="wide")

#assumes user has specific GSE they want to look at
gse_id = st.text_input("Enter GSE Series ID", "")

# get webscraping results
geo = pd.read_csv('geo_webscrap.csv')

# check that user entered GSE ID in the csv
if gse_id in geo['Series'].values:
    gse = geo[geo['Series'] == gse_id]
else:
    st.error("Please enter a valid GSE ID")
    st.stop()

# get the summary and overall design information from the csv
text = [gse['Summary'].iloc[0], gse['Overall design'].iloc[0]]
text = " ".join(text)

# let user chose spacy model
options = ["en_ner_bionlp13cg_md", "en_ner_bc5cdr_md", "en_ner_jnlpba_md"]

spacy_model = st.selectbox(
    "Select an NER model:",  # Label for the dropdown
    options             # List of options
)

st.title("Summary Entity Annotation")
text = st.text_area("Text to analyze", text, height=200)

clean_text = text.replace('\n', ' ')
doc = spacy_streamlit.process_text(spacy_model, clean_text)

# define the labels detected in the text
labels: Sequence[str] = tuple()
labels = labels or list({ent.label_ for ent in doc.ents})

if len(labels) == 0:
    st.error("No text assigned to entities available in this model. Please select a different model")
    st.stop()
else:
    # Create distict colours for labels
    col_dict = {}
    s_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    for label, colour in zip(labels, s_colours):
        col_dict[label] = colour
    
    # visualize the NER data
    spacy_streamlit.visualize_ner(
        doc,
        show_table=False,
        title="Select entities to visualize",
        colors=col_dict,
    )
    st.text(f"Analyzed using spaCy model {spacy_model}")