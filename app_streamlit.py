import streamlit as st
import socket
import os
import openpyxl
import shutil
import re
import string
import json
import PyPDF2
import io
import Levenshtein
import openai
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
import threading
import fitz
import pandas as pd
import pytesseract
from PyPDF2 import PdfReader
import csv
from urllib.parse import quote, unquote
from pdf2image import convert_from_path
from PIL import Image
import tabula
from streamlit_functions import set_favicon
from streamlit_functions import failed_page
from manual_test import (
    statement_to_xlsx,
    count_exported_csv_files,
    delete_exported_csv_files,
    extract_numbers_from_string,
    delete_temp_files,
    statement_to_csv,
    get_exported_files,
    run_ocr_to_csv,
    run_ocr_to_csv_multiple_times,
    improve_text_structure,
    generate_explanation,
    delete_thumbnails,
    pdf_to_csv_conversion,
    extract_text_from_page,
    convert_page_to_image,
    process_pdf,
)
import ast
import PyPDF2
import signal
import sys
import os
from utils import (
    count_exported_csv_files,
    delete_thumbnails,
    get_exported_files
)
from zipfile import ZipFile
import threading

# File upload directory
UPLOAD_FOLDER = os.path.join('static', 'img_photo')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist


openai.api_key = os.environ['api_key']

# Get current directory
current_dir = os.getcwd()


# Initialize session state variables for persistence
if 'partners' not in st.session_state:
    st.session_state['partners'] = {}

if 'global_excel_file' not in st.session_state:
    st.session_state['global_excel_file'] = None

if 'exported_files' not in st.session_state:
    st.session_state['exported_files'] = None

# load dot_env should've been here



def are_you_sure_page():

    st.markdown("<h3>Files are ready to be transferred to the main xlsx file. Please make sure to check if everything is in order.</h3>", unsafe_allow_html=True)

    output_dir = os.path.join(os.getcwd(), 'output') # Get current working directory for deployment

    # Get download links for files in output directory
    count = count_exported_csv_files(output_dir)
    download_links = {}
    for file_index in range(count):
        file_name = f'exported{file_index}.csv'
        download_links[file_name] = f"/download/{file_name}"

    # Delete thumbnails
    delete_thumbnails()

    # Get list of exported files
    exported_files = get_exported_files()

    # Sort download links for consistent order
    sorted_download_links = dict(sorted(download_links.items()))
    
    # Display checkboxes for file selection and download links
    st.markdown("<h4>Select files to include:</h4>", unsafe_allow_html=True)
    selected_files = []
    for file, download_link in sorted_download_links.items():
        if st.checkbox(file, key=file):
            selected_files.append(file)
            st.markdown(f"- <a href='{download_link}' download>{file}</a>", unsafe_allow_html=True)
    
    if st.button("Convert to CSV Again"):
        # Store selected files in session state for use in other parts of your app
        st.session_state['selected_files'] = selected_files
        st.experimental_rerun()  # Trigger a page refresh to update UI
        
        # Redirect to the "/run_extract" page (you'll need to implement this in Streamlit)
        st.experimental_set_query_params(page="/run_extract")

    if st.button("I'm Done, Thanks"):
        # Redirect to the "/go_back" page (you'll need to implement this in Streamlit)
        st.experimental_set_query_params(page="/go_back")


def download_file(filename):
    """
    Function to facilitate file downloads in Streamlit.
    
    Args:
        filename (str): The name of the file to be downloaded.
    """

    output_dir = os.path.join(os.getcwd(), 'output')  # Get the current working directory
    file_path = os.path.join(output_dir, filename)

    # Check if the file exists
    if not os.path.isfile(file_path):
        st.error(f"File '{filename}' not found.")
        return

    # Serve the file to the user for download
    with open(file_path, "rb") as f:
        bytes_data = f.read()
        st.download_button(
            label=f"Download {filename}",
            data=bytes_data,
            file_name=filename,
            mime="text/csv",  # Adjust MIME type as needed
        )

def run_extract_page():
    """
    Streamlit function to handle re-extraction of selected files.
    """
    global partners
    global exported_files
    each_results_per_page = {}
    if "selected_files" not in st.session_state:
        st.warning("No files have been selected for extraction. Please navigate back to the previous page and select files.")
        return

    selected_files = st.session_state.get("selected_files", [])

    if not selected_files:
        st.warning("No files have been selected for re-extraction.")
        return

    with st.spinner("Re-extracting selected files..."):
        for filename in selected_files:
            original_image_text = pytesseract.image_to_string(Image.open(partners[filename]))
            page_number = extract_numbers_from_string(filename)
            temp_page_number = extract_numbers_from_string(partners[filename])
            
            results = run_ocr_to_csv_multiple_times(
                original_image_text, temp_page_number, num_iterations=6
            )
            each_results_per_page[page_number] = results[1]

        for page_number, results in each_results_per_page.items():
            statement_to_csv(results['csv_data'], page_number)

    are_you_sure()
    st.experimental_rerun()


def run_again_page():
    """
    Streamlit function to handle re-running the extraction process with updated terms.
    """
    st.subheader("Enter Terms with Alternatives")

    # Text area for user input
    textarea_content = st.text_area(
        "Enter terms with their alternatives separated by line breaks:",
        value="{\n    'total interest bearing liabilities or financial debt':['financial debt','total liabilities'],\n    # ... (Add other terms and alternatives here)\n}",
        height=300,
    )

    if st.button("Extract to Excel"):
        try:
            # Parse the dictionary from the text area content
            start_index = textarea_content.find("{")
            end_index = textarea_content.rfind("}")
            dictionary_str = textarea_content[start_index : end_index + 1]
            my_dict = ast.literal_eval(dictionary_str.strip())

            # Get the current working directory for deployment compatibility
            current_dir = os.getcwd()

            # Count CSV files in the output directory
            count = count_exported_csv_files(os.path.join(current_dir, "output"))

            # Transfer extracted data to Excel
            the_url = transfer_to_excel(count, my_dict)

            # Delete temporary files
            delete_exported_csv_files(os.path.join(current_dir, "output"))
            delete_temp_files()

            # Copy the generated Excel file to the output directory
            source_file = 'uploads/new_version.xlsx'
            destination_file = os.path.join(current_dir, 'output', 'new_version.xlsx')
            shutil.copyfile(source_file, destination_file)
            
            st.markdown(f"""
            ### Extraction Complete!
            Your Excel file is ready for download: <a href="{the_url}" download>Download Excel</a>
            """, unsafe_allow_html=True)
        except (SyntaxError, ValueError):
            st.error("Invalid dictionary format. Please check your input.")


def go_back_page():
    """
    Streamlit function to handle navigating back to the home page.
    """
    
    # Delete files
    current_dir = os.getcwd()
    delete_exported_csv_files(os.path.join(current_dir, 'output'))
    delete_temp_files()
    delete_thumbnails()
    

    # Redirect to the home page (replace 'index' with the actual page name)
    st.experimental_set_query_params(page="index") 
    st.experimental_rerun()  # Refresh the app to load the new page



def index_page():
    """
    Streamlit function to handle the index page.
    """

    global global_excel_file
    global selected_results
    
    # Initialize session state variables for the first run
    if 'pdf_text' not in st.session_state:
        st.session_state['pdf_text'] = None
    if 'search_words' not in st.session_state:
        st.session_state['search_words'] = None
    
    st.markdown("<h1 class='mt-5'>PDF to CSV converter</h1>", unsafe_allow_html=True)
    st.markdown("<h1 class='mt-5'>Bank Statement PDF to Excel</h1>", unsafe_allow_html=True)

    with st.form("searchForm"):
        # Input widgets for file uploads
        first_file = st.file_uploader("Upload First PDF:", type="pdf", key="first_file")
        second_file = st.file_uploader("Upload Second PDF:", type="pdf", key="second_file")
        excel_file = st.file_uploader("Upload excel file (to send data to):", type=["xlsx", "xls"], key="excel_file")
        
        # Search words input
        search_words = st.text_input("Search Words:")
        
        # Submit button
        if st.form_submit_button("Search"):
            if first_file:
                
                global_excel_file = os.path.join("output", "new_version.xlsx")
                

                if search_words:
                    st.session_state['search_words'] = search_words
                    with st.spinner("Processing..."):
                        st.session_state['pdf_text'] = process_pdf(
                            first_file, second_file, search_words
                        )
                    # Reset selected results
                    selected_results = []
                else:
                    st.error("Please enter search words.")
            else:
                st.error("No PDF file uploaded.")

    # Display results
    if st.session_state['pdf_text']:
        with st.form("extractForm"):
            st.markdown(f"<h2>Search Results for '{st.session_state['search_words']}':</h2>", unsafe_allow_html=True)

            selected_results = []
            for result in st.session_state['pdf_text']:
                cols = st.columns([1, 2, 1, 5, 2])  # Adjust column ratios as needed
                with cols[0]:
                    # Use the unique page number as the checkbox key
                    selected = st.checkbox(
                        "", key=f"checkbox_{result['page_number']}", value=False
                    )
                    if selected:
                        selected_results.append(result)

                # Display the thumbnail in a smaller column
                with cols[1]:
                    st.image(result["thumbnail_path"], width=100)
                with cols[2]:
                    st.write(result['page_number']) 
                with cols[3]:
                    st.write(result['text'])
                with cols[4]:
                    st.write(result['explanation'])
        
            if st.form_submit_button("Extract CSV"):
                if selected_results:
                    # Pass the selected results to the next stage
                    st.session_state['selected_results'] = selected_results
                    # Redirect to the extraction page
                    st.experimental_set_query_params(page="/extract") 
                    st.experimental_rerun()  # Refresh the app to load the new page


def view_pdf_page():
    """
    Streamlit function to handle bank statement PDF viewing and processing.
    """
    
    global partners
    each_results_per_page_b = []
    
    # File uploader for the bank statement PDF
    bs_file = st.file_uploader("Upload Bank Statement (PDF):", type="pdf")
    
    if bs_file is not None:
        # Create a temporary file path for the uploaded PDF
        temp_pdf_path = "temp.pdf"
        
        # Save the uploaded file temporarily
        with open(temp_pdf_path, "wb") as f:
            f.write(bs_file.read())

        # Process the PDF
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(temp_pdf_path)
            num_pages = len(pdf_reader.pages)

            for i in range(num_pages):
                # Extract image from the PDF page
                image = convert_from_path(temp_pdf_path, first_page=i, last_page=i, single_file=True)[0]
                
                # Save the image temporarily
                image_path = f"bank_statement_page{i}.png"
                image.save(image_path)
                
                # Perform OCR on the image
                original_image_text = pytesseract.image_to_string(Image.open(image_path))
                
                # Update global partners dictionary
                partners['exported'+str(i)+'.csv'] = image_path

                # Run OCR to CSV multiple times
                results = run_ocr_to_csv_multiple_times(original_image_text, i, num_iterations=6)
                
                # Store results
                each_results_per_page_b.append(results[1])
                
            # Convert extracted data to CSV
            for x in range(len(each_results_per_page_b)):
                statement_to_csv(each_results_per_page_b[x]['csv_data'], x)
            
            os.remove(temp_pdf_path)  # Clean up the temporary PDF file
        
        # Redirect to are_you_sure page
        st.session_state['each_results_per_page_b'] = each_results_per_page_b  # Store results for later
        are_you_sure_page()


def extract_csv_page():
    """
    Streamlit function to handle CSV extraction from selected results.
    """
    global partners
    each_results_per_page = []
    image_paths = []

    if 'selected_results' not in st.session_state or not st.session_state['selected_results']:
        st.error("No results selected for CSV extraction. Please go back and select results.")
        return

    with st.spinner("Extracting CSV files..."):
        for result in st.session_state['selected_results']:
            page_number = result['page_number']
            image_path = os.path.join(os.getcwd(), f"temp_page_{page_number}.png")
            image_paths.append(image_path)

            original_image_text = pytesseract.image_to_string(Image.open(image_path))
            partners[f"exported{len(image_paths) - 1}.csv"] = image_path

            results = run_ocr_to_csv_multiple_times(original_image_text, page_number, num_iterations=6)
            each_results_per_page.append(results[1])
        
        # Clear selected_results from session state
        del st.session_state['selected_results']

        for x in range(len(each_results_per_page)):
            statement_to_csv(each_results_per_page[x]["csv_data"], x)

    are_you_sure_page()  # Call the are_you_sure function after extraction
    st.experimental_rerun()


def transfer_to_excel_page():
    """
    Streamlit function to handle transferring CSV data to an Excel file.
    """
    # Retrieve necessary data from the session state
    each_results_per_page = st.session_state.get('each_results_per_page_b', [])  # Use 'each_results_per_page_b' from session
    dictionary = st.session_state.get('dictionary', {})

    if not each_results_per_page:
        st.error("No CSV files found for transfer to Excel. Please go back and extract data.")
        return

    with st.spinner("Transferring to Excel..."):
        current_dir = os.getcwd()  # Get the current working directory for deployment

        for x in range(len(each_results_per_page)):
            try:
                csv_file_path = os.path.join(current_dir, "output", f"exported{x}.csv")
                with open(csv_file_path, "r") as file:
                    csv_data = file.read()
                    the_url = statement_to_xlsx(csv_data, global_excel_file, dictionary)
            except FileNotFoundError:
                st.error(f"CSV file not found: {csv_file_path}")
                return
            except Exception as e:  # Catch any other potential errors
                st.error(f"An error occurred during Excel transfer: {e}")
                return failed()

    st.success("CSV extraction and Excel transfer successful!")
    st.markdown(f"""
    ### Download Your Excel File: <a href="{the_url}" download>Download Excel</a>
    """, unsafe_allow_html=True)


def serve_image(filename):
    """
    Streamlit function to serve images from the 'static/images' directory.
    
    Args:
        filename (str): The name of the image file.
    """

    image_path = os.path.join(os.getcwd(), 'static', 'images', filename)  # Adjust path based on your project structure

    if not os.path.isfile(image_path):
        st.error(f"Image '{filename}' not found.")
        return

    with open(image_path, "rb") as f:
        image_data = f.read()
        st.image(image_data, caption=filename)


