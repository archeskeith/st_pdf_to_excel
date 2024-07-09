import streamlit as st
from PIL import Image
# import fitz
# from fitz import *
import pdfplumber
# from fitz_fix.fitz import * 
from PyPDF2 import PdfMerger
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import boto3
import tempfile
import time
from concurrent.futures import as_completed
import pytesseract
from PIL import Image
import openai
import base64
from io import StringIO
import csv
import ocrmypdf
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# get the directory where the script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR,'app_data')
# print(f"Base Directory: {BASE_DIR}")

OUTPUT_DIR = os.path.join(BASE_DIR,"output")
STATIC_DIR = os.path.join(BASE_DIR,"static")
UPLOADS_DIR = os.path.join(BASE_DIR,"uploads")
THUMBNAILS_DIR = os.path.join(STATIC_DIR,"thumbnails")
# construct paths relative to the base directory (create if they don't exist yet)
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR,"static"),exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "thumbnails"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

GLOBAL_EXCEL_FILE_URL = None
# current_dir = os.getcwd()

def update_selected_pages(page_num):
    if page_num in st.session_state.selected_results:
        st.session_state.selected_results.remove(page_num)
        index = st.session_state.selected_results.index(page_num)
        st.session_state.selected_csv_data.pop(index)
    else:
        result_index = [result['page_number'] + 1 for result in st.session_state['pdf_text']].index(page_num)
        st.session_state.selected_results.append(page_num)
        st.session_state.selected_csv_data.append(st.session_state['pdf_text'][result_index]['text'])
    

def string_to_csv(input_string):
    """Converts a string with '|' and '\n' delimiters into CSV format."""

    # Initialize empty lists for rows and headers
    csv_rows = []
    headers = []

    # Create a file-like object from the string for csv.reader
    csv_stringio = StringIO(input_string)
    csv_reader = csv.reader(csv_stringio, delimiter='|')

    # Iterate through the rows in the string
    for i, row in enumerate(csv_reader):
        # Remove extra spaces, remove commas, and convert to int for numeric values
        # cleaned_row = [entry.strip().replace(',', '') for entry in row]
        cleaned_row = [entry.strip().replace(",",'') if isinstance(entry,str) else entry for entry in row] 

        if i == 0:
            # The first row is considered the header
            headers = cleaned_row
        else:
            # For subsequent rows, try to convert numeric values to integers
            try:
                cleaned_row = [int(x) if x.isdigit() else x for x in cleaned_row]
            except ValueError:
                pass  # If not numeric, keep it as a string

            # Append the cleaned row to the list of rows
            csv_rows.append(cleaned_row)

    for x in headers:
        print(x)
        print(type(x))
    # Remove the newline in headers and rows
    # headers = [header.replace('\n', '') for header in headers]
    # csv_rows = [[cell.replace('\n', '') for cell in row] for row in csv_rows]
    
    # Remove '\n' from headers and rows before returning (only for string values)
    headers = [header.replace('/n', '') if isinstance(header, str) else header for header in headers]
    csv_rows = [[cell.replace('/n', '') if isinstance(cell, str) else cell for cell in row] for row in csv_rows]

    return headers, csv_rows


# def extract_text_from_page(page):
#     # Extract text and thumbnail from a PDF page (try to extract text directly first)
#     text = page.get_text()
#     if text.strip() == "":  # If no text is extracted (likely an image-based pdf)
#         pix = page.get_pixmap()
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         try:
#             text = pytesseract.image_to_string(img)
#         except pytesseract.TesseractError as e:
#             st.error(f"OCR Error on page {page.number}:{e}")
#             text = ""

#     pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
#     thumbnail_path = os.path.join(THUMBNAILS_DIR, f'page_{page.number}_thumbnail.jpg')
#     pix.save(thumbnail_path)

#     return {
#         'page_number': page.number,
#         'text': text,
#         'explanation': 'Explanation placeholder',
#         'thumbnail_path': f'static/thumbnails/page_{page.number}_thumbnail.jpg',
#     }


def extract_text_from_page(page_num, pdf_path):
    """Extracts text and thumbnail from a PDF page using pdfplumber."""

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        
        # Try table extraction first
        tables = page.extract_tables()
        if tables:
            st.write(f"Tables found on page {page_num + 1}")
            return {
                'page_number': page_num,
                'text': "",  # Or you can include some table summary here
                'tables': tables,
                'thumbnail_path': f'static/thumbnails/page_{page_num + 1}_thumbnail.png',
            }

        # If no tables found, fall back to text extraction (OCR if needed)
        text = page.extract_text(x_tolerance=3, y_tolerance=3, use_text_flow=False, text_layout=True)

    # Create thumbnail
    im = page.to_image(resolution=150)
    thumbnail_path = os.path.join(THUMBNAILS_DIR, f'page_{page_num + 1}_thumbnail.png')
    im.save(thumbnail_path, format="PNG") 

    return {
        'page_number': page_num,
        'text': text,
        'explanation': 'Explanation placeholder',
        'thumbnail_path': f'static/thumbnails/page_{page_num + 1}_thumbnail.png', 
    }


def extract_table_with_openai(result, model="gpt-4o"):
    """Extracts tabular data using OpenAI API for a given page number within the search results."""

    page_num = result["page_number"]
    thumbnail_path = result["thumbnail_path"]

    image_path = os.path.join(BASE_DIR, thumbnail_path)

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")  # Encode to base64

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You will be provided with an image of a table. Please extract the data from the table and provide it in CSV format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please extract the data from the table in the image and provide it in CSV format. Copy as much as possible the text provided, but add in | (for new columns) and /n (for new rows), according to how it looks like on the image. Indicate '/n' if a new row is seen. Make sure years are directly aligned with the columns they are placed at (IMPORTANT). Do it directly, no need to respond, just do the table. Make sure the output really looks like the table shown in the picture"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ]
        )
        extracted_table_data = response["choices"][0]["message"]["content"]
        if extracted_table_data == "No table found":
            st.warning(f"No table found on page {result['page_number'] + 1}")
            return None

        # Normalize and Clean the extracted data
        extracted_table_data = extracted_table_data.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove empty values from the table
        extracted_table_data = "\n".join([line for line in extracted_table_data.splitlines() if line.strip()])
        
        # print(extracted_table_data)
        headers, csv_rows = string_to_csv(extracted_table_data)
        st.write(headers)
        st.write(csv_rows)

        # Write the CSV file (add header first, then rows)
        with open(
            os.path.join(OUTPUT_DIR, f"page_{page_num + 1}.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(csv_rows)

        st.success(f"CSV File page_{page_num + 1}.csv has been extracted.")
        return f"page_{page_num + 1}.csv"  # This should be the returned file name
        # return pd.read_csv(StringIO(extracted_table_data), lineterminator='\n')
        # return extracted_table_data
    except openai.error.OpenAIError as e:  # Moved here
        st.error(f"OpenAI API Error on page {page_num + 1}: {e}")
        return None
    except pd.errors.EmptyDataError:  # Catch empty CSV data
        st.warning(f"Empty table on page {page_num + 1}")
        return None

def search_pdfs(pdf_files, search_words, excel_file=None):
    output_dir = os.path.join(BASE_DIR, "output")
    # global_excel_file = os.path.join(output_dir, "new_version.xlsx")

    # Merge PDFs (if multiple files have been uploaded)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=BASE_DIR) as temp_pdf:
        if len(pdf_files) > 1:
            merger = PdfMerger()
            for pdf in pdf_files:
                merger.append(pdf)
            merger.write(temp_pdf)
        else:
            temp_pdf.write(pdf_files[0].read())
        temp_pdf_path = temp_pdf.name

        # OCR the temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=BASE_DIR) as ocr_temp_pdf:
            ocrmypdf.ocr(temp_pdf_path, ocr_temp_pdf.name, deskew=True)  # Perform OCR

    # Open the OCR'd PDF file
    with pdfplumber.open(ocr_temp_pdf.name) as pdf:
        # Search logic using ThreadPoolExecutor (parallelism)
        results = []
        search_words_list = search_words.lower().split()

        with ThreadPoolExecutor() as executor:
            # Use page_num (0-based indexing) directly in pdfplumber
            futures = [executor.submit(extract_text_from_page, page_num, ocr_temp_pdf.name) 
                        for page_num in range(len(pdf.pages))] 

            for future in as_completed(futures):
                result = future.result()
                result_text = result['text'].lower()
                if not search_words_list or any(word in result_text for word in search_words_list):
                    results.append(result)
                    # print(result)


    # Sort results by page number (add 1 to page_number since it's 0-based in pdfplumber)
    results.sort(key=lambda x: x['page_number'] + 1)

    # Remove the temporary PDF files
    os.remove(temp_pdf_path)
    os.remove(ocr_temp_pdf.name)  # Remove the OCR'd PDF file
    return results  
# def search_pdfs(pdf_files,search_words,excel_file=None):
#     output_dir = os.path.join(BASE_DIR,"output")
#     global_excel_file = os.path.join(output_dir, "new_version.xlsx")
    
#     # merge PDFs (if multiple files has been uploaded)
#     with tempfile.NamedTemporaryFile(suffix='.pdf',delete=False,dir=BASE_DIR) as temp_pdf:
#         if len(pdf_files) > 1:
#             merger = PdfMerger()
#             for pdf in pdf_files:
#                 merger.append(pdf)
#             merger.write(temp_pdf)
#         else:
#             temp_pdf.write(pdf_files[0].read())
        
#         temp_pdf_path = temp_pdf.name
    
#     # open merged pdf or single pdf
#     doc = fitz.open(temp_pdf_path)

#     # search logic using ThreadPoolExecutor (parallelism)
#     results = []
#     search_words_list = search_words.lower().split()

#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(extract_text_from_page,page.number - 1, temp_pdf_path) for page in doc.pages()]
        
#         for future in as_completed(futures):
#             result = future.result()
#             result_text = result['text'].lower()
#             if not search_words_list or any(word in result_text for word in search_words_list):
#                 result['thumbnail_path'] = os.path.join("thumbnails", os.path.basename(result['thumbnail_path']))
#                 results.append(result)
    
#     results.sort(key=lambda x: x['page_number'])
#     doc.close()
#     os.remove(temp_pdf_path)
#     # st.write(results)
#     return results



# streamlit app
st.title("PDF to CSV Converter")
st.subheader("Search PDFs")

# initialization session state variables
if "selected_results" not in st.session_state:
    st.session_state.selected_results = []
if "selected_csv_data" not in st.session_state:
    st.session_state.selected_csv_data = []
if "pdf_text" not in st.session_state:
    st.session_state['pdf_text'] = None


# add a flag in the session state to track button clicks
# if 'extract_clicked' not in st.session_state:
#     st.session_state.extract_clicked = False

with st.form('searchForm'):
    pdf_files = st.file_uploader("Upload PDF Files:",accept_multiple_files=True)
    excel_file = st.file_uploader("Upload Excel File:",type=['xlsx','xls'],key="excel_file")

    # using same names as in the original code
    search_words = st.text_input("Search Words:")
    

    if st.form_submit_button("Search"):
        if not pdf_files:
            st.warning("Please upload at least one PDF file.")
        else:
            try:
                pdf_text = search_pdfs(pdf_files, search_words, excel_file)
                st.session_state['pdf_text'] = pdf_text

                if pdf_text:
                    st.subheader(f"Search Results for '{search_words}:")

                    # Initialize variables to store selected page number
                    # selected_results = st.session_state["selected_results"] # Access from session state
                    # selected_csv_data = st.session_state["selected_csv_data"]

                    # Loop through the results
                    for i, result in enumerate(pdf_text):
                        with st.container():
                            st.write(f"Page {result['page_number'] + 1}:")
                            col1,col2 = st.columns([2,8])

                            with col1:
                                st.write(f"Page {result['page_number'] + 1}")
                                
                            # Image and text in the second column
                            with col2:
                                image = Image.open(os.path.join(BASE_DIR, result['thumbnail_path']))
                                st.image(image, caption=f"Page {result['page_number']+ 1} Thumbnail", use_column_width=True)
                      
                    
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")
        if excel_file is not None:
            try:
                save_path = os.path.join(UPLOADS_DIR,excel_file.name)
                with open(save_path,"wb") as f:
                    f.write(excel_file.read())
                GLOBAL_EXCEL_FILE_URL = f"/uploads/{excel_file.name}"
            except Exception as e:
                st.error(f"Error uploading Excel file: {e}")


# Add a new form for page selection
with st.form('extractionForm'):
    st.write("Enter Page Numbers (comma-separated), e.g. '1,2,3':")
    selected_pages_input = st.text_input("Pages", "")
    
    # Submit button for CSV extraction
    if st.form_submit_button("Extract Selected to CSV"):
        if not st.session_state.get("pdf_text"):  # Make sure pdf_text is available
            st.warning("Please search for PDFs first.")
        else:
            pdf_text = st.session_state['pdf_text']
            try:
                # Split and clean the input, allowing for spaces and empty entries
                selected_pages = []
                for x in selected_pages_input.split(","):
                    stripped_x = x.strip()
                    if stripped_x:  # Check if it's not empty after stripping
                        try:
                            selected_pages.append(int(stripped_x) - 1)
                        except ValueError:
                            st.warning(f"Ignoring invalid page input: {x}")

                # Use set to remove duplicates and convert back to list
                selected_pages = list(set(selected_pages))
                
                extracted_csv_data = []
                for page_num in selected_pages:
                    csv_filename = f"page_{page_num + 1}.csv"  # Create filename based on page number
                    if 0 <= page_num < len(pdf_text):  # Check if page number is valid
                        # Get the result corresponding to the page_num
                        result = next((r for r in pdf_text if r["page_number"] == page_num), None)
                        
                        if result is not None:  # Check if the result was found
                            table_df = extract_table_with_openai(result)  # Pass the whole result dictionary
                            if table_df is not None:
                                extracted_csv_data.append(table_df)
                            else:
                                text = pdf_text[page_num]['text']  # Access text from pdf_text
                                with open(csv_filename, 'w', encoding='utf-8') as f:  # Open file in write mode
                                    f.write(text)  # Directly write the text to the CSV file
                                extracted_csv_data.append(text) # Append the text to extracted_csv_data as well
                                st.success(f"CSV File '{csv_filename}' has been extracted (from text).")
                    else:
                        st.warning(f"Invalid page number: {page_num + 1}")
                        continue
                
                if len(extracted_csv_data) > 0:
                    # Check if all elements in extracted_csv_data are strings
                    if all(isinstance(item, str) for item in extracted_csv_data):
                        # If they are, create a DataFrame with a single column named "Text"
                        final_df = pd.DataFrame({"Text": extracted_csv_data})
                    else:
                        # Otherwise, concatenate the DataFrames
                        final_df = pd.concat(extracted_csv_data, ignore_index=True)
                    
                    csv_filename = "extracted.csv"
                    final_df.to_csv(csv_filename, index=False)
                    st.success(f"CSV File '{csv_filename}' has been extracted.")
                else:
                    st.info("No tables found in selected pages.")  # Updated message

            except Exception as e:  # Catch more general exceptions
                st.error(f"Error extracting CSV: {e}")
