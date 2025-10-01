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

# Create values from 1 to 1000 in steps of 100
options = list(range(1, 251, 50))

with colY:
    st.title(":dna: GEO Search")
    st.write("The GEO Search app dashboard using streamlit.")
    # example
    search_term = "lung cancer spatial"

    text_search = st.text_input(
        "",
        value=search_term,
        placeholder="Type the search keyword and press enter ... ",
    )

    retmax = st.select_slider(
        "**Choose the number of records to retrieve**",
        options=options,
        value=options[0]  # default = 1
    )

if text_search:
    st.markdown(f"### Searching for `{text_search}` ...")
    res_count, results = fetch_geo_datasets(text_search, api_key, email, retmax)
    st.markdown(f'##### Found `{res_count}` datasets, showing `{retmax}`.')
    summary = [result["summary"] for result in results]
    summary_df = pd.DataFrame(summary)
    summary_df['Samples'] = summary_df['Samples'].apply(lambda x: ', '.join([s.get('Title') for s in x]))
    summary_df["details"] = summary_df["Id"].apply(lambda x: f"./details?acc={x}")
    print(summary_df.Samples)
    st.dataframe(summary_df, hide_index=True, column_config={"Item": None, "Id": None, "GSE": None, 
                "entryType": None, "ptechType": None, "valType": None, "SSInfo": None, "subsetInfo": None, 
                "Relations": None, "ExtRelations": None, "SeriesTitle": None, "PlatformTitle": None, "PlatformTaxa": None, "SamplesTaxa": None,
                "Projects": None, "FTPLink": None, "GDS": None, "summary": None, "GEO2R": None,
                "details": st.column_config.LinkColumn(
                    "Details",
                    display_text="View",
                    help="Click to see details"
                )})
