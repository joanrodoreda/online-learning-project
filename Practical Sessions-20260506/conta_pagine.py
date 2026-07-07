from pathlib import Path
from pypdf import PdfReader

cartella = Path(r"C:\Users\davbo\Desktop\ola\Nuova cartella")

totale_pagine = 0
totale_pdf = 0

for pdf in cartella.glob("*.pdf"):
    try:
        pagine = len(PdfReader(pdf).pages)
        totale_pagine += pagine
        totale_pdf += 1
        print(f"{pdf.name}: {pagine} pagine")
    except Exception as e:
        print(f"Errore con {pdf.name}: {e}")

print("\n--- RIEPILOGO ---")
print(f"PDF trovati: {totale_pdf}")
print(f"Totale pagine: {totale_pagine}")