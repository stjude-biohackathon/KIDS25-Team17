from utils.entrez_search import *
import os
from dotenv import load_dotenv

#load the env variables
load_dotenv()
api_key = os.getenv("API_KEY")
email = os.getenv("EMAIL")

Entrez.api_key = api_key
Entrez.email = email


# Fetch the platform record
handle = Entrez.esummary(db="gds", term="GPL24247")
record = Entrez.read(handle)

platform_name = record[0]['title']
print("Platform Name:", platform_name)
