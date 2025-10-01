import requests

#get the gse folder name
def gse_folder(gse_id):
    _gse = gse_id[:6]+'nnn'
    return _gse

#gse ftp links
#https://ftp.ncbi.nlm.nih.gov/geo/series/GSE267nnn/GSE267960/suppl/filelist.txt

def format_size_convert(bytes_size:int):
    KB = 1024
    MB = KB * 1024
    GB = MB * 1024

    if bytes_size >= GB:
        size = bytes_size / GB
        unit = "GB"
    elif bytes_size >= MB:
        size = bytes_size / MB
        unit = "MB"
    elif bytes_size >= KB:
        size = bytes_size / KB
        unit = "KB"
    else:
        size = bytes_size
        unit = "Bytes"
    
    return f"{size:.2f} {unit}"


def gse_supp_list(gse_id, timeout):
    
    base_url = 'https://ftp.ncbi.nlm.nih.gov/geo/series/'
    url = f'{base_url}{gse_folder(gse_id)}/{gse_id}/suppl/filelist.txt'
    ##print(url)

    try:
        response = requests.get(url, timeout = timeout)
        response.raise_for_status()
        data = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

    file_details = []
    lines = data.splitlines()
    
    for l in lines:
        file_dict = {}
        if l.startswith('#'):
            continue
        l = l.split('\t')
        if l[0] == 'Archive':
            file_dict['file_name'] = l[1]
            file_dict['type'] = "Main archive"
            file_dict['size'] = format_size_convert(int(l[3]))
        else:
            file_dict['file_name'] = l[1]
            file_dict['type'] = "File"
            file_dict['size'] = format_size_convert(int(l[3]))

        file_details.append(file_dict)
    return file_details

#print(gse_supp_list('GSE268048', 2))

