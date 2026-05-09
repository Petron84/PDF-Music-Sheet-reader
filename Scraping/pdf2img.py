from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pypdfium2 as pdfium
import os


def pdf_request():
    Tk().withdraw()

    pdf_path = askopenfilename(
        title = "Please Select a music PDF",
        filetypes = [("PDF Files", "*.pdf")]
    )
    return pdf_path
    
def pdf_convert(pdf_path):

    folder = "media"
    os.makedirs(folder, exist_ok=True)

    save_path = os.path.join(
        folder,
        f"page_.png"
          )

    pdf = pdfium.PdfDocument(pdf_path)

    for i, page in enumerate(pdf):
         image = page.render(scale=4).to_pil()
         image.save(save_path)
         break
    print("Image Converted")

pdf_request()