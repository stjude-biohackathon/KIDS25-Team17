import streamlit as st
import pandas as pd
from utils.entrez_search import *
from custom_files import *
import os
from dotenv import load_dotenv
import spacy_streamlit
import spacy
from typing import Sequence

# access query parameters
query_params = st.query_params
gse_id = query_params["acc"]

#load the env variables
load_dotenv("../")
api_key = os.getenv("API_KEY")
email = os.getenv("EMAIL")


# Enable GPU support for spaCy
#spacy.prefer_gpu()
spacy_model = "en_ner_bionlp13cg_md"

st.set_page_config( page_title=f"Details of {gse_id}", page_icon=":dna:", layout="wide")

if gse_id:
    details = fetch_esummary(gse_id, api_key, email)
    acc = details['Accession']
    pubs = details['PubMedIds'][0].numerator
    desc = str(details['summary'])
    doc = spacy_streamlit.process_text(spacy_model, desc)
    
    #define the labels detected in the text
    labels: Sequence[str] = tuple()
    labels = labels or list({ent.label_ for ent in doc.ents})

    # Create distict colours for labels
    col_dict = {}
    s_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
    for label, colour in zip(labels, s_colours):
        col_dict[label] = colour

    supp_files = gse_supp_list(acc, 2)
    supp_df = pd.DataFrame(supp_files)

colX, colY, colZ = st.columns([1,10,1])

with colY:
    st.title(f"Details of `{acc}`")
    st.write(f"#### Study title: {details['title']}")
    st.write(f"#### Organism: `{details['taxon']}`")
    st.write(f"#### Published date: `{details['PDAT']}`")
    st.write(f"#### Associated publication: `PMID{pubs}`")
    st.write(f"#### Sample no : `{details['n_samples'].numerator}`")
    st.write(f"#### Sample names : `{', '.join([x.get('Title') for x in details['Samples']])}`")
    st.write(f"#### Download folder : `{details['FTPLink']}`")

    #visualize the NER data
    spacy_streamlit.visualize_ner(
        doc,
        # labels=["PERSON", "DATE", "GPE"],
        show_table=False,
        title="Select entities to visualize",
        colors=col_dict,
    )
    st.write("#### Supplementary files")
    st.dataframe(supp_df)

