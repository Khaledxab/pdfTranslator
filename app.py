import PyPDF2
from fpdf import FPDF
from transformers import MarianMTModel, MarianTokenizer

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def translate_text(text, target_language, max_length=512):
    """
    Translates text to the target language.
    """
    model_name = f'Helsinki-NLP/opus-mt-{target_language}-en'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    text_segments = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    translated_segments = []
    for segment in text_segments:
        inputs = tokenizer(segment, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated_segments.append(translated_text)

    translated_text = " ".join(translated_segments)
    return translated_text


def create_pdf_from_text(text, output_path):
    """
    Creates a PDF file from text.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 5, txt=text, align="L")
    pdf.output(output_path)

input_pdf_path = 'input.pdf'
output_pdf_path = 'output.pdf'
target_language = 'fr'

text = extract_text_from_pdf(input_pdf_path)
translated_text = translate_text(text, target_language)
create_pdf_from_text(translated_text, output_pdf_path)
