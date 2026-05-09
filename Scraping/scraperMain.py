import scraper 
from sites import sites
import time
import pdf2img

def crawlMain():
    site = sites[1]
    base_url = site["url"]
    webpage = scraper.get_webpage(base_url)
    links = scraper.get_links(webpage, base_url)
    pdf_links = scraper.follow_links(site, links)

def downloadMain():
    pdf = pdf2img.pdf_request()
    pdf2img.pdf_convert(pdf)


downloadMain()