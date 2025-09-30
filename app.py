import streamlit as st
import pandas as pd
from utils.entrez_search import *
import os
from dotenv import load_dotenv

#load the env variables
load_dotenv()
api_key = os.getenv("API_KEY")
email = os.getenv("EMAIL")

st.set_page_config( page_title="GEO Search", page_icon=":dna:", layout="wide")

colX, colY, colZ = st.columns([3,6,3])

with colY:
    st.title(":dna: GEO Search")
    st.write("The GEO Search app dashboard using streamlit.")
    # example
    search_term = "lung cancer spatial"

    text_search = st.text_input(
        "",
        value="",
        placeholder="Type the search keyword and press enter ... ",
    )

if text_search:
    st.markdown(f"### Searching for `{text_search}` ...")
    results = fetch_geo_datasets(text_search, api_key, email)
    summary = [result["summary"] for result in results]
    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df, hide_index=True, column_config={"Item": None, "Id": None, "GDS": None, "summary": None, "GEO2R": None})
