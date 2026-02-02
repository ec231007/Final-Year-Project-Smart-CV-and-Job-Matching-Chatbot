import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # "text" layout preserves some structure
    return text

# Test it with one of your sample PDFs
if __name__ == "__main__":
    test_resume = "data/Unprocessed_cv/data/ENGINEERING/10030015.pdf"
    try:
        content = extract_text_from_pdf(test_resume)
        print("Successfully extracted text!")
        print(content[:500]) # Print first 500 chars
    except Exception as e:
        print(f"Error: {e}")