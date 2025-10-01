from Bio import Entrez


def fetch_geo_datasets(search_term, api_key, email, retmax: int = 100):
    # use api key avoid rate limiting
    #retmax limit is 100000

    Entrez.api_key = api_key
    Entrez.email = email
    handle = Entrez.esearch(db="gds", term=search_term, retmode="xml", usehistory="y", retmax = retmax)
    record = Entrez.read(handle)
    handle.close()

    print(f"Total records found: {record["Count"]}")

    out_list = []
    # find details for each gse id
    if record["IdList"]:
        for gse_id in record["IdList"]:
            out_dict = {}
            handle = Entrez.esummary(db="gds", id=gse_id, retmode="xml", usehistory="y")
            summary = Entrez.read(handle)
            handle.close()
            summary = summary[0]
            out_dict["id"] = gse_id
            out_dict["summary"] = summary
            out_list.append(out_dict)
        return record["Count"], out_list

#just get esummary
def fetch_esummary(gse_id, api_key, email):
    
    Entrez.api_key = api_key
    Entrez.email = email     
    
    handle = Entrez.esummary(db="gds", id=gse_id, retmode="xml", usehistory="y")
    summary = Entrez.read(handle)
    handle.close()
    summary = summary[0]

    return summary