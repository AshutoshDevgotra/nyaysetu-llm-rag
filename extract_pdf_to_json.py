import fitz  # PyMuPDF
import json
from pathlib import Path

output = []
pdf_folder = Path("pdfs")

print("📂 PDF folder exists:", pdf_folder.exists())
pdf_files = list(pdf_folder.glob("*.pdf"))
print("📄 PDF files found:", len(pdf_files))

for pdf_file in pdf_files:
    print(f"🔍 Reading {pdf_file.name}")
    doc = fitz.open(pdf_file)
    full_text = ""
    for i, page in enumerate(doc):
        print(f"   🧾 Page {i+1}")
        full_text += page.get_text() + "\n"
    if full_text.strip():
        output.append({"filename": pdf_file.name, "text": full_text.strip()})
    else:
        print(f"⚠️ No extractable text in {pdf_file.name}")
    doc.close()

# Save to JSON
with open("nyaysetu_pdf_texts.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(output)} PDFs to nyaysetu_pdf_texts.json")
