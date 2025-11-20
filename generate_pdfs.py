import os
from fpdf import FPDF

os.makedirs('sample_pdfs', exist_ok=True)

pdf_contents = [
    ["Sample PDF 1", "This is a test PDF about Python.", "It is used for testing the RAG pipeline."],
    ["Sample PDF 2", "Another test document.", "This file contains facts about AI.", "This should help during retrieval."],
    ["Sample PDF 3", "RAG stands for Retrieval-Augmented Generation.", "Here is a third sample with generic information."]
]

for idx, content in enumerate(pdf_contents, 1):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content:
        pdf.cell(200, 10, txt=line, ln=1)
    pdf.output(f"sample_pdfs/sample{idx}.pdf")

print("Generated 3 sample PDFs in the 'sample_pdfs/' directory.")
