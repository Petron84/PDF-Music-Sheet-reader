import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import os

def get_webpage(url):
    print(url)
    # Grabing the htlm from the website and making sure we get a response
    response = requests.get(url)
    response.raise_for_status()
    # Converting to beautiful soup object
    time.sleep(1)
    return BeautifulSoup(response.content, "html.parser")




def get_links(webpage, base_url):
    links = []

    for tag in webpage.find_all("a", href=True):
        href = tag.get("href")
        full_url = urljoin(base_url, href)
        links.append(full_url)
    
    return links

def load_set(filename):
    try:
        with open(filename, "r") as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()


def filter_links(links, keyword=None, file_type=None):
    results = []

    for link in links:
        if keyword and keyword not in link: 
            continue

        if file_type and not link.endswith(file_type):
            continue

        results.append(link)
    
    return(results)

def follow_links(site, links):

    name = site["name"]
    start_url = site["url"]

    folder = f"data/{name}"
    os.makedirs(folder, exist_ok=True)

    pdf_path = f"{folder}/pdf_links.txt"
    visited_path = f"{folder}/visited_urls.txt"

    #Setting sets :)
    links_searched = 0
    pdfs_found = 0
    visited = load_set(visited_path)
    pdf_links = load_set(pdf_path)
    to_visit = list(links)

    #Timing the crawler
    start_time = time.time()
    time_limit = 300

    pdf_file = open(pdf_path, "a", encoding="utf-8")
    visited_file = open(visited_path, "a", encoding="utf-8")



    while to_visit:
        if time.time() - start_time > time_limit:
            print("Time limit reached. Stopping crawl.")
            break


        url = to_visit.pop()
  
        if url in visited:
            print(f"{url} already visited")
            continue

        visited.add(url)
        links_searched += 1

        visited_file.write(url + "\n")
        visited_file.flush()

        try:
            page = get_webpage(url)
        except:
            print(f"Couldn't reach {url}")
            continue

        new_links = get_links(page, url)

        for link in new_links:
            if link.endswith(".pdf") and "-a4" not in link:
                print(f"found pdf {link}")
                pdf_links.add(link)
                pdfs_found += 1

                pdf_file.write(link + "\n")
                pdf_file.flush()
            
            elif start_url in link and link not in visited:
                to_visit.append(link)
    
    print(f"Found {pdfs_found} pdf's and searched {links_searched} links!")
    pdf_file.close()
    visited_file.close()
    
    return pdf_links




def download_links(links):
    for link in links:
        print(f"Downloading {link}")
        pdf = requests.get(link).content

            