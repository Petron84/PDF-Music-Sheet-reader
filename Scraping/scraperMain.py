from Scraping import scraper 
from Scraping import sites
import time
from Scraping import pdf2img

def crawlMain():
    site = sites[1]
    base_url = site["url"]
    webpage = scraper.get_webpage(base_url)
    links = scraper.get_links(webpage, base_url)
    pdf_links = scraper.follow_links(site, links)

def downloadsheet():
    pdf = pdf2img.pdf_request()
    pdf2img.pdf_convert(pdf)


