# PDF Translation Tool

## Overview
This Python project is a PDF translation tool that leverages pre-trained machine translation models for language conversion and text extraction. It can extract text from PDF files, translate it to a target language, and generate a new translated PDF document.

## Features
- Extracts text from PDF files using PyPDF2.
- Translates extracted text to a target language using pre-trained Transformers models.
- Creates a new PDF document with the translated text using FPDF.

## Usage
1. Ensure you have the necessary dependencies installed, including PyPDF2, Transformers, FPDF, and PyTorch for GPU support.
2. Provide your input PDF file by replacing 'input.pdf' in the code.
3. Specify the target language for translation by modifying the 'target_language' variable.
4. Run the code to perform the PDF translation.

## Dependencies
- PyPDF2
- Transformers
- FPDF
- PyTorch (for GPU support)

## Installation
You can install the required dependencies using pip:
```shell
pip install -r requirements.txt
