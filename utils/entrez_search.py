from Bio import Entrez


def fetch_geo_datasets(search_term, api_key, email):
    # use api key avoid rate limiting
    Entrez.api_key = api_key
    Entrez.email = email
    handle = Entrez.esearch(db="gds", term=search_term, retmode="xml")
    record = Entrez.read(handle)
    handle.close()

    out_list = []
    # find details for each gse id
    if record["IdList"]:
        for gse_id in record["IdList"]:
            out_dict = {}
            print(f"Found GSE ID: {gse_id}")
            handle = Entrez.esummary(db="gds", id=gse_id, retmode="xml")
            summary = Entrez.read(handle)
            handle.close()
            summary = summary[0]
            out_dict["id"] = gse_id
            out_dict["summary"] = summary
            out_list.append(out_dict)
        return out_list
