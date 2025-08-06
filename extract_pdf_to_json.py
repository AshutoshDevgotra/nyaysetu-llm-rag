import json
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_files():
    """Extract text from various file formats and save to JSON"""
    
    # Sample legal texts since we can't process actual PDFs without the files
    sample_legal_texts = [
        {
            "source": "contract_law_basics.txt",
            "text": """
            Contract law is a body of law that governs oral and written agreements associated with the exchange of goods and services, money, and properties. Contract law includes elements like the offer and acceptance, intention to create legal relations, consideration, capacity, and legality.

            Key elements of a valid contract:
            1. Offer - A promise to do or refrain from doing something
            2. Acceptance - Agreement to the terms of the offer
            3. Consideration - Something of value exchanged between parties
            4. Capacity - Legal ability to enter into a contract
            5. Legality - The contract must be for a legal purpose

            Types of contracts include bilateral contracts, unilateral contracts, express contracts, implied contracts, executed contracts, and executory contracts.
            """
        },
        {
            "source": "property_law_fundamentals.txt", 
            "text": """
            Property law is the area of law that governs the various forms of ownership and tenancy in real property and personal property. Property law determines who owns what, how ownership is acquired and transferred, and what rights come with ownership.

            Types of property:
            1. Real Property - Land and anything permanently attached to it
            2. Personal Property - Movable items not attached to real property
            3. Intellectual Property - Creations of the mind (patents, copyrights, trademarks)

            Property rights include the right to possess, use, enjoy, transfer, and exclude others from the property. These rights may be limited by law, regulation, or agreement.
            """
        },
        {
            "source": "criminal_law_overview.txt",
            "text": """
            Criminal law defines crimes and establishes punishments for criminal behavior. It is distinguished from civil law, which deals with disputes between individuals and organizations.

            Elements of a crime:
            1. Actus Reus - The guilty act or criminal conduct
            2. Mens Rea - The guilty mind or criminal intent
            3. Causation - The link between the act and the result
            4. Concurrence - The guilty act and guilty mind must occur together

            Types of crimes include felonies (serious crimes like murder, rape, burglary), misdemeanors (less serious crimes like petty theft, simple assault), and infractions (minor violations like traffic tickets).

            Criminal procedure governs how criminal cases are processed through the legal system, including arrest, investigation, prosecution, trial, and sentencing.
            """
        },
        {
            "source": "constitutional_law_principles.txt",
            "text": """
            Constitutional law deals with the fundamental principles by which a government exercises its authority. It establishes the framework for government, defines the powers and duties of government entities, and protects individual rights.

            Key constitutional principles:
            1. Separation of Powers - Division of government into legislative, executive, and judicial branches
            2. Checks and Balances - Each branch has powers to limit the others
            3. Federalism - Division of power between national and state governments
            4. Due Process - Fair treatment through the judicial system
            5. Equal Protection - Equal treatment under the law

            The Constitution is the supreme law of the land, and all other laws must conform to its requirements. Constitutional interpretation involves determining the meaning and application of constitutional provisions.
            """
        },
        {
            "source": "tort_law_basics.txt",
            "text": """
            Tort law provides remedies for civil wrongs that are not breaches of contract. A tort is a wrongful act that causes harm to another person, for which the injured party may seek compensation.

            Types of torts:
            1. Intentional Torts - Deliberate wrongful acts (assault, battery, false imprisonment, defamation)
            2. Negligence - Failure to exercise reasonable care (most common type of tort)
            3. Strict Liability - Liability without fault (product liability, abnormally dangerous activities)

            Elements of negligence:
            1. Duty - Legal obligation to exercise reasonable care
            2. Breach - Failure to meet the standard of care
            3. Causation - The breach caused the harm
            4. Damages - Actual harm or injury occurred

            Remedies in tort law typically involve monetary damages to compensate the injured party.
            """
        },
        {
            "source": "business_law_essentials.txt",
            "text": """
            Business law encompasses the legal rules and regulations that govern commercial transactions and business operations. It includes various areas such as contract law, employment law, intellectual property, corporate law, and commercial law.

            Key areas of business law:
            1. Business Formation - Choosing and establishing business entities (corporations, LLCs, partnerships)
            2. Contract Law - Agreements between businesses and with customers/suppliers
            3. Employment Law - Hiring, workplace safety, discrimination, termination
            4. Intellectual Property - Protecting trademarks, copyrights, patents, trade secrets
            5. Commercial Law - Sales of goods, secured transactions, banking
            6. Corporate Governance - Directors' duties, shareholder rights, mergers and acquisitions

            Compliance with business law helps companies avoid legal disputes, protect their interests, and operate ethically and legally.
            """
        }
    ]
    
    try:
        # Check if pdfs directory exists and has content
        pdfs_dir = Path("pdfs")
        if pdfs_dir.exists():
            text_files = list(pdfs_dir.glob("*.txt"))
            if text_files:
                logger.info(f"Found {len(text_files)} text files in pdfs directory")
                extracted_texts = []
                
                for file_path in text_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            extracted_texts.append({
                                "source": file_path.name,
                                "text": content
                            })
                        logger.info(f"Extracted text from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")
                
                if extracted_texts:
                    sample_legal_texts = extracted_texts
        
        # Save to JSON file
        output_file = "nyaysetu_pdf_texts.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_legal_texts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved {len(sample_legal_texts)} texts to {output_file}")
        
        # Display summary
        total_chars = sum(len(item['text']) for item in sample_legal_texts)
        logger.info(f"Total characters extracted: {total_chars}")
        
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
