import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # "text" layout preserves some structure
    return text

def extract_text_from_word(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['docx', 'doc']:
        return extract_text_from_word(file_path)
    else:
        return ""

# Test it with one sample PDFs
if __name__ == "__main__":
    test_resume = "data/Unprocessed_cv/data/ENGINEERING/10030015.pdf"
    try:
        content = extract_text_from_pdf(test_resume)
        print("Successfully extracted text!")
        print(content[:500]) # Print first 500 chars
    except Exception as e:
        print(f"Error: {e}")