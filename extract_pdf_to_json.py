import json
import os
import logging
from pathlib import Path
import PyPDF2
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")
                    continue

            return text.strip()

    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return None

def extract_text_from_files():
    """Extract text from PDF files and save to JSON"""

    extracted_texts = []
    
    try:
        # Check if pdfs directory exists and has content
        pdfs_dir = Path("pdfs")
        if pdfs_dir.exists():
            # Process PDF files
            pdf_files = list(pdfs_dir.glob("*.pdf"))
            if pdf_files:
                logger.info(f"Found {len(pdf_files)} PDF files in pdfs directory")

                for file_path in pdf_files:
                    logger.info(f"Processing {file_path.name}...")
                    text_content = extract_text_from_pdf(file_path)

                    if text_content and len(text_content.strip()) > 50:  # Only include PDFs with substantial content
                        extracted_texts.append({
                            "source": file_path.name,
                            "text": text_content
                        })
                        logger.info(f"Successfully extracted {len(text_content)} characters from {file_path.name}")
                    else:
                        logger.warning(f"No substantial text extracted from {file_path.name}")

            # Also process any text files as backup
            text_files = list(pdfs_dir.glob("*.txt"))
            if text_files:
                logger.info(f"Found {len(text_files)} text files in pdfs directory")

                for file_path in text_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) > 50:
                                extracted_texts.append({
                                    "source": file_path.name,
                                    "text": content
                                })
                        logger.info(f"Extracted text from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")

        # If no files were processed, use sample data as fallback
        if not extracted_texts:
            logger.warning("No PDF or text files processed, using sample legal texts as fallback")
            extracted_texts = [
                {
                    "source": "sample_contract_law.txt",
                    "text": "Contract law governs agreements between parties. Key elements include offer, acceptance, consideration, capacity, and legality."
                }
            ]
        
        # Save to JSON file
        output_file = "nyaysetu_pdf_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_texts, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(extracted_texts)} texts to {output_file}")

        # Display summary
        total_chars = sum(len(item['text']) for item in extracted_texts)
        logger.info(f"Total characters extracted: {total_chars}")

        # Log sources for verification
        sources = [item['source'] for item in extracted_texts]
        logger.info(f"Processed sources: {sources}")

        return True
        
    except Exception as e:
        logger.error(f"Error in text extraction: {e}")
        return False

if __name__ == "__main__":
    success = extract_text_from_files()
    if success:
        print("✅ Text extraction completed successfully!")
    else:
        print("❌ Text extraction failed")
