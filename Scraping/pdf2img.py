from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pypdfium2 as pdfium
import os
from PIL import Image


def pdf_request():
    Tk().withdraw()

    pdf_path = askopenfilename(
        title = "Please Select a music PDF",
          filetypes=[
        ("Supported Files", "*.pdf *.png"),
        ("PDF Files", "*.pdf"),
        ("PNG Files", "*.png"),]
    )
    return pdf_path
    
def pdf_convert(pdf_path):
     dpi=500
     folder = "media"
     os.makedirs(folder, exist_ok=True)
    
     save_path = os.path.join(
        folder,
        f"selected_sheet.png"
          )
     
     #Checks if the selected file is already a PNG image, if so it just saves it to the media folder without conversion
     if pdf_path.lower().endswith('.png'):
        print("Selected file is already a PNG image.")
        image = Image.open(pdf_path)
        image.save(save_path)
        return


     pdf = pdfium.PdfDocument(pdf_path)

     for i, page in enumerate(pdf):
         image = page.render(scale=dpi/72).to_pil()
         image.save(save_path)
         break
     print("Image Converted")
