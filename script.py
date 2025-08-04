import fitz  # PyMuPDF
import os

def extract_text_from_all_pdfs(folder_path):
    all_text_data = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                doc = fitz.open(file_path)
                full_text = ""

                for page in doc:
                    full_text += page.get_text()

                all_text_data.append({
                    "filename": filename,
                    "text": full_text.strip()
                })

                print(f"✅ Extracted: {filename}")

            except Exception as e:
                print(f"❌ Skipped {filename}: {e}")

    return all_text_data

# Use relative path to pdfs folder
pdf_folder = "pdfs"

documents = extract_text_from_all_pdfs(pdf_folder)

# Save output to JSON
import json
with open("nyaysetu_pdf_texts.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Extracted text from {len(documents)} PDFs.")
