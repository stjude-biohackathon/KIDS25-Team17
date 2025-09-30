import scispacy
import spacy
from Bio import Entrez

#nlp = spacy.load("en_ner_bionlp13cg_md")

def fetch_pubmed_text(pmid, email):
    Entrez.email = email
    
    # Step 1: Get article summary and look for PMC ID
    summary_handle = Entrez.esummary(db="pubmed", id=str(pmid), retmode="xml")
    summary = Entrez.read(summary_handle)
    summary_handle.close()
    
    try:
        article_ids = summary[0]['ArticleIds']
        pmc_id = None
        for aid in article_ids:
            if aid.attributes['IdType'] == 'pmc':
                pmc_id = aid.title()
                break
    except Exception:
        pmc_id = None
    
    # Step 2: If PMC ID exists, try to get full text in XML
    if pmc_id:
        try:
            pmc_handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
            pmc_fulltext = pmc_handle.read()
            pmc_handle.close()
            if pmc_fulltext.strip():
                return pmc_fulltext  # This will be in XML format
        except Exception:
            pass  # Try to fetch abstract if full text retrieval fails
    
    # Step 3: Fetch PubMed record and abstract
    try:
        pubmed_handle = Entrez.efetch(db="pubmed", id=str(pmid), rettype="xml", retmode="xml")
        records = Entrez.read(pubmed_handle)
        pubmed_handle.close()
        article = records['PubmedArticle'][0]
        abstract_text = article['MedlineCitation']['Article']['Abstract']['AbstractText']
        if isinstance(abstract_text, list):
            return "\n".join(str(text) for text in abstract_text)
        else:
            return str(abstract_text)
    except Exception:
        return "No full text or abstract available for this PMID in PubMed or PMC."