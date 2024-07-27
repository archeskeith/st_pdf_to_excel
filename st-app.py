import streamlit as st
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer,LTFigure
import pytesseract
from PIL import Image
# import fitz
# from fitz import *
import pdfplumber
# from fitz_fix.fitz import * 
from pypdf import PdfWriter
from pdf2image import convert_from_path  
# from PyPDF2 import PdfMerger
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import boto3
import tempfile
import time
from concurrent.futures import as_completed
from PIL import Image
import openai
from openai import OpenAI # Add this line of code
import base64
from io import StringIO
import csv
import ocrmypdf
from pdfminer.layout import LTChar
from pdf2image import convert_from_path
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# st.write(f"pdf2image version: {pdf2image.__version__}")  
# get the directory where the script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR,'app_data')
# print(f"Base Directory: {BASE_DIR}")

OUTPUT_DIR = os.path.join(BASE_DIR,"output")
STATIC_DIR = os.path.join(BASE_DIR,"static")
UPLOADS_DIR = os.path.join(BASE_DIR,"uploads")
THUMBNAILS_DIR = os.path.join(STATIC_DIR, "thumbnails") 
# construct paths relative to the base directory (create if they don't exist yet)
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR,"static"),exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "thumbnails"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
os.environ["POPPLER_PATH"] = "/home/linuxbrew/.linuxbrew/bin/pdfinfo"

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
        cleaned_row = [entry.strip().replace(",","") if isinstance(entry,str) else entry for entry in row] 

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

    # for x in headers:
    #     # print(x)
    #     # print(type(x))
    # Remove the newline in headers and rows
    # headers = [header.replace('\n', '') for header in headers]
    # csv_rows = [[cell.replace('\n', '') for cell in row] for row in csv_rows]
    
    # Remove '\n' from headers and rows before returning (only for string values)
    headers = [header.replace('/n', '') if isinstance(header, str) else header for header in headers]
    csv_rows = [[cell.replace('/n', '') if isinstance(cell, str) else cell for cell in row] for row in csv_rows]
    print(headers)
    print(csv_rows)
    return headers, csv_rows

def remove_newlines(headers, rows):
    """Removes newline characters (\n or /n) from headers and rows in a CSV structure, even if within numbers."""
    def clean_cell(cell):
        """Helper function to clean a single cell."""
        if isinstance(cell, str):
            return cell.replace("\n", "").replace("/", "")
        return cell

    # Clean headers
    clean_headers = [clean_cell(header) for header in headers]

    # Clean rows (and try converting to numbers after cleaning)
    clean_rows = []
    for row in rows:
        processed_row = [clean_cell(cell) for cell in row]
        try:
            processed_row = [
                int(cell.replace(",", "")) if isinstance(cell, str) and cell.replace(",", "").replace(".", "").isdigit() 
                else cell
                for cell in processed_row
            ]
        except ValueError:
            pass  # Keep the cell as a string if conversion fails
        clean_rows.append(processed_row)

    return clean_headers, clean_rows



def final_string_to_csv(input_string):
    """Converts a string with '|' and '\n' delimiters into CSV format.

    This function removes commas from numeric values, handles newlines, and
    removes newline characters from string values.
    """
    rows = input_string.splitlines()

    # Preprocess rows to remove commas from numeric values and '\n' from strings
    processed_rows = []
    for row in rows:
        columns = row.split("|")
        processed_columns = []
        for col in columns:
            if isinstance(col, str):
                col = col.replace('\n', '')  # Remove newlines from strings
                try:
                    processed_columns.append(int(col.replace(",", "")))  # Remove commas and convert to integer
                except ValueError:
                    processed_columns.append(col)  # Not an integer, keep as string
            else:
                processed_columns.append(col)  # If not a string, keep the original value

        processed_rows.append(processed_columns)

    headers = processed_rows[0]  # Extract headers
    rows = processed_rows[1:]  # Extract data rows

    return headers, rows

def extract_text_from_page(page_num, pdf_path):
    """Extracts text and thumbnail from a PDF page using pdfminer.six and pdf2image."""
    

    with open(pdf_path, 'rb') as file:
        for page_layout in extract_pages(file, page_numbers=[page_num]):
            text = ""
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text += element.get_text()
            break  # Exit the loop after processing the desired page

    # Create thumbnail using pdf2image
    
    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
    
    thumbnail_path = os.path.join(THUMBNAILS_DIR, f'page_{page_num + 1}_thumbnail.png')
    images[0].save(thumbnail_path, "PNG")

    # If no text, try OCR
    if not text.strip():
        text = pytesseract.image_to_string(images[0])

    return {
        'page_number': page_num,
        'text': text,
        'thumbnail_path': f'static/thumbnails/page_{page_num + 1}_thumbnail.png', 
    }


def extract_table_with_openai(result, model="gpt-4o-mini"):
    """Extracts tabular data using OpenAI API for a given page number."""
    # st.write("RESULT: ",result)
    # print('try')
    page_num = int(result["page_number"])
    # print('obanai')

    thumbnail_path = result["thumbnail_path"]
    image_path = os.path.join(BASE_DIR, thumbnail_path)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare the image data as base64 string
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can extract data from tables in images."},
                {
                    "role": "user",
                    "content": (
                        "Please extract the data from the following text and provide it in CSV format. Copy as much as possible the text provided, but add in |"
                        "(for new columns) and \\n (for new rows), according to how it looks like in the text. Indicate '\\n' if a new row is seen."
                        "Make sure years are directly aligned with the columns they are placed at (IMPORTANT)."
                        "Do it directly, no need to respond, just do the table."
                        "Make sure the output really looks like the table shown in the text.\n\n" 
                        + str(result['text']) # Convert the result to string 
                    )
                },
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )
        # print(response)
        # st.write(response.choices[0].message.content)
        message_content = response.choices[0].message.content
        # print("MUICHIRO ",message_content)
        st.write(message_content)
        st.write(string_to_csv(message_content))
        headers, rows = final_string_to_csv(message_content)
        headers, rows = remove_newlines(headers, rows) 
        # Write the CSV file (add header first, then rows)
        filename = f"page_{page_num + 1}.csv"
        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)


        # print(response.choices[0])
        return None
        # return(response.choices[0].messsage.content)
    # Error handling
    except openai.OpenAIError as e:
        st.error(f"OpenAI API Error on page {page_num + 1}: {e}")
        return None
    except FileNotFoundError as e:
        st.error(f"File Not Found Error on page {page_num + 1}: {e}")
        return None
    except pd.errors.EmptyDataError:
        st.warning(f"Empty table on page {page_num + 1}")
        return None



# def search_pdfs(pdf_files, search_words, excel_file=None):
#     output_dir = os.path.join(BASE_DIR, "output")

#     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=BASE_DIR) as temp_pdf:
#         if len(pdf_files) > 1:
#             merger = PdfMerger()
#             for pdf in pdf_files:
#                 merger.append(pdf)
#             merger.write(temp_pdf)
#         else:
#             temp_pdf.write(pdf_files[0].read())
#         temp_pdf_path = temp_pdf.name

#     with pdfplumber.open(temp_pdf_path) as pdf:
#         results = []
#         search_words_list = search_words.lower().split()

#         for page_num in range(len(pdf.pages)):
#             result = extract_text_from_page(page_num, temp_pdf_path)  
#             result_text = result['text'].lower()
#             if not search_words_list or any(word in result_text for word in search_words_list):
#                 results.append(result)

#     results.sort(key=lambda x: x['page_number'] + 1)

#     # Remove the temporary PDF files
#     os.remove(temp_pdf_path)

#     return results  

def merge_pdfs(pdf_files, output_path):
    """Merges multiple PDF files into a single PDF."""
    merger = PdfWriter()

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)  
        for page in reader.pages:
            merger.add_page(page)

    # Write the merged PDF to the output path
    with open(output_path, "wb") as output_pdf:
        merger.write(output_pdf)

def search_pdfs(pdf_files, search_words, excel_file=None):
    """Searches for keywords in uploaded PDF files and returns results."""

    # Merge PDFs if necessary
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, dir=BASE_DIR) as temp_pdf:
        if len(pdf_files) > 1:
            merge_pdfs(pdf_files, temp_pdf.name)  # Merge using PdfWriter
        else:
            temp_pdf.write(pdf_files[0].read())
        temp_pdf_path = temp_pdf.name

    # Extract text and thumbnails from pages
    with pdfplumber.open(temp_pdf_path) as pdf:
        results = []
        search_words_list = search_words.lower().split()

        for page_num in range(len(pdf.pages)):
            result = extract_text_from_page(page_num, temp_pdf_path)
            result_text = result['text'].lower()
            if not search_words_list or any(word in result_text for word in search_words_list):
                results.append(result)

    results.sort(key=lambda x: x['page_number'] + 1)

    # Remove the temporary PDF file
    os.remove(temp_pdf_path)

    return results



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
                csv_filenames = []
                for page_num in selected_pages:
                    csv_filename = f"page_{page_num + 1}.csv"  # Create filename based on page number
                    csv_filenames.append(csv_filename)
                    if 0 <= page_num < len(pdf_text):  # Check if page number is valid
                        # Get the result corresponding to the page_num
                        result = next((r for r in pdf_text if r["page_number"] == page_num), None)
                        
                        if result is not None:  # Check if the result was found
                            # print("RESULT: ",result['text'])
                            # table_df = extract_table_with_openai(result['text']) 

                            extract_table_with_openai(result)
                            # print('TABLE: ',table_df)
                             # Pass the whole result dictionary
                            # st.write(table_df)
                            # if table_df is not None:
                            #     extracted_csv_data.append(table_df)
                            # else:
                            #     text = pdf_text[page_num]['text']  # Access text from pdf_text
                            #     with open(csv_filename, 'w', encoding='utf-8') as f:  # Open file in write mode
                            #         f.write(text)  # Directly write the text to the CSV file
                            #     extracted_csv_data.append(text) # Append the text to extracted_csv_data as well
                                # if csv_filename:
                                #     
                                #     st.success(f"CSV File '{csv_filename}' has been extracted (from text).")

                                #     # After successful extraction, set the flag to show the download button
                                #     st.session_state['show_download_button'] = True
                                #     st.session_state['csv_filenames'] = csv_filename
                    else:
                        st.warning(f"Invalid page number: {page_num + 1}")
                        continue
                if len(csv_filenames)>0:
                    st.success(f"CSV File '{csv_filename}' has been extracted (from text).")
                    # After successful extraction, set the flag to show the download button
                    st.session_state['show_download_button'] = True
                    st.session_state['csv_filenames'] = csv_filenames
                if len(extracted_csv_data) > 0:
                    # Check if all elements in extracted_csv_data are strings
                    if all(isinstance(item, str) for item in extracted_csv_data):
                        # If they are, create a DataFrame with a single column named "Text"
                        final_df = pd.DataFrame({"Text": extracted_csv_data})
                    else:
                        # Otherwise, concatenate the DataFrames
                        final_df = pd.concat(extracted_csv_data, ignore_index=True)
                    
                    csv_filename = f"page_{page_num + 1}.csv"
                    final_df.to_csv(csv_filename, index=False)
                    st.success(f"CSV File '{csv_filename}' has been extracted.")
                else:
                    st.info("No tables found in selected pages.")  # Updated message

            except Exception as e:  # Catch more general exceptions
                st.write('it passed here')
                st.error(f"Error extracting CSV: {e}")

if st.session_state.get("show_download_button",False):
    number_of_pages = st.session_state["csv_filenames"]
    for x in number_of_pages:
        the_filename = x
        with open(x,"rb") as file:
            st.download_button(
                label=x,
                data=file,
                file_name = x,
                mime="text/csv"
            )


# Clear Everything Form
with st.form('clearForm'):
    if st.form_submit_button("CLEAR EVERYTHING"):
        # Clear relevant session state variables
        st.session_state.selected_results = []
        st.session_state.selected_csv_data = []
        st.session_state['pdf_text'] = None
        st.session_state['show_download_buttons'] = []
        st.session_state['extracted_csvs'] = {}

        # Clear output directory
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.error(f"Error clearing output directory: {e}")
        st.success("Everything cleared successfully!")
        # Rerun the app to refresh the UI (optional, but recommended for clarity)
        st.experimental_rerun()
