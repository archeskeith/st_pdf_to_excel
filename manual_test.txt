import os
from flask import Flask, send_file,url_for
# from dotenv import load_dotenv
from werkzeug.datastructures import FileStorage
from openpyxl import Workbook, load_workbook
import re 
import shutil
import tempfile
import string
import concurrent.futures
import json
import signal
import PyPDF2
from io import BytesIO
import Levenshtein  
import openai
from openpyxl.utils import get_column_letter
from openpyxl.utils import column_index_from_string
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfMerger
import pytesseract
import fitz
import csv
import camelot
from openpyxl import Workbook

# from current chatgpt4 openapi API key access
# open.api_key = "${{ secrets.OPENAI_KEY }}"
# open.api_key = "${{ creds.api_key }}"
# open.api_key = os.environ['api_key']
openai.api_key = os.environ['api_key']
# print(os.environ['api_key'])
current_dir = os.getcwd()

# count the number of pages a pdf has
def count_pages(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.numPages        
        return num_pages

def extract_tables_from_pdf(pdf_path):
    # extract tables from the PDF using camelot
    tables = camelot.read_pdf(pdf_path, flavor='stream', pages='1-end')
    return tables

def write_tables_to_excel(tables, excel_file):
    wb = Workbook()
    ws = wb.active
    
    for table in tables:
        df = table.df
        for row in df.iterrows():
            ws.append(row[1].tolist())
    
    wb.save(excel_file)

def process_pdf(file, second_file,search_words):
    
    if (second_file):
        merger = PdfMerger()
        merger.append(file)
        merger.append(second_file)
        # print("FILE: ",file)
        # # Save the merged PDF to a temporary file
        temp_name = os.path.join(current_dir,"merged_pdf.pdf")
        merger.write(temp_name)
        merger.close()
        doc = fitz.open("merged_pdf.pdf")
    else:
        temp_name = os.path.join(current_dir,"merged_pdf.pdf")
        file.save(temp_name)

    doc = fitz.open("merged_pdf.pdf")

    result = []

    ssearch_words = search_words
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_text_from_page, page,"merged_pdf.pdf") for page in doc.pages()]
        
        search_words = search_words.split()

        for future in futures:
            result_text = future.result()['text'].lower()
            if any(word.lower() in result_text for word in search_words):
                # any words in search_words are present in the result text
                result.append(future.result())
    doc.close()

    return result

def convert_page_to_image(page):
    # converting a non-image page to an image
    try:
        image = page.to_image(resolution=600)  
        return Image.frombytes("RGB", [image.width, image.height], image.samples)
    except Exception as e:
        print(f"Error converting page to image: {e}")
        return None

def extract_text_from_page(page,pdf_path):

    pixmap = (page).get_pixmap()
    if pixmap is None:
        page = convert_page_to_image(page)
        pixmap = (page).get_pixmap()
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    image_text = pytesseract.image_to_string(image)
    
    pdf_to_csv_conversion(pdf_path, page.number+1)
    explanation = generate_explanation(image_text)

    # generate thumbnail (for image previews)
    pixmap = page.get_pixmap()
    
    thumbnail = image.resize((300, 300))
    
    thumbnail_path = os.path.join(current_dir, "static", "thumbnail_page_" + str(page.number+1) + ".png")

    # static folder is needed for flask
    thumbnail_name = '../static/thumbnail_page_'+str(page.number+1)+'.png'
    thumbnail.save(thumbnail_path)
    
    return {'page_number': page.number + 1, 'text': image_text, 'explanation': explanation, 'thumbnail_path': thumbnail_name}

# for a clean directory
def delete_thumbnails():
    folder_path = "static"
    pattern = "thumbnail_page_"

    # iterate through the files in the directory
    for filename in os.listdir(folder_path):

        # check if there are filenames with thumbnail_page_<number>.png
        if filename.startswith(pattern) and filename.endswith(".png"):

            file_path = os.path.join(folder_path, filename)

            # deleting the file
            os.remove(file_path)

def generate_explanation(text):
    # use chatgpt to generate explanation for text 
    prompt = f"Explain the contents of the following text:\n{text}"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    explanation = response.choices[0].text.strip()
    return explanation

def improve_text_structure(text):
    # improving text structure using chatgpt 4 prompt
    prompt = f"Improve the structure and formatting of the following text:\n{text}"
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.7
    )
    enhanced_text = response.choices[0].text.strip()
    return enhanced_text

def improve_table_structure(table_text):
    # use chatgpt 4 to improve structuring (prompts)
    prompt = f"Improve the structure and formatting of the following table:\n{table_text}"
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.7
    )
    enhanced_table = response.choices[0].text.strip()
    return enhanced_table

def pdf_to_csv_conversion(pdf_file, page_number, num_iterations=5):
    # accessing pdf and extracting each page
    with open(os.path.join(current_dir, pdf_file), 'rb') as file:
        pdf_reader = PdfReader(file)

        # converting pdf to image
        images = convert_from_path(file.name, first_page=page_number, last_page=page_number, single_file=True)
        print("IMAGES: ",images)
        print("FILE.NAME: ",file.name)
        
        image = images[0]

        image_path = os.path.join(current_dir, f"temp_page_{page_number}.png")
        image.save(image_path)

def run_ocr_to_csv_multiple_times(original_image_text, page_number, num_iterations=5):
    results = []
    image_path = os.path.join(current_dir, f"temp_page_{page_number}.png")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit OCR tasks to the executor
        ocr_futures = [executor.submit(run_ocr_to_csv, original_image_text, page_number) for _ in range(num_iterations)]

        # Collect results from OCR tasks
        for future in concurrent.futures.as_completed(ocr_futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Handle exceptions if any
                print(f"Error during OCR: {e}")

    # Find the best result based on Levenshtein distance
    best_result = min(results, key=lambda x: Levenshtein.distance(x[0], original_image_text))
    return best_result[0], best_result[1], image_path


def run_ocr_to_csv(original_image_text, page_number):
    model_name = 'gpt-4-vision-preview'
    image_path = os.path.join(current_dir, f"temp_page_{page_number}.png")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Convert the following OCR text to CSV:\n{original_image_text}. Copy as much as possible the text provided, but add in | (for new columns) and /n (for new rows), according to how it looks like on the image: {Image.open(image_path)}. Indicate '/n' if a new row is seen, based on the pdf image :{Image.open(image_path)}. Indicate a | if a new column is seen, based on the pdf image {Image.open(image_path)}. Make sure years are directly aligned with the columns they are placed at (IMPORTANT). Do it directly, no need to respond, just do the table."}
    ]
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    if 'choices' in response and response['choices']:
        csv_data = response['choices'][0]['message']['content']
        return csv_data, {'engine': model_name, 'csv_data': csv_data}
    else:
        return f"Error: {response}", None

def get_exported_files():
    output_dir = os.path.join(os.getcwd(), 'output')
    exported_files = [file for file in os.listdir(output_dir) if file.startswith("exported") and file.endswith(".csv")]
    return exported_files

def statement_to_csv(input_statement,page):
    # formatting

    # remove commas
    input_statement = input_statement.replace(",", "")

    # replace indicators '|' and '/n' with appropriate separators
    cleaned_statement = input_statement.replace('|', ',').replace('/n', '\n')

    # splitting into lines
    lines = cleaned_statement.split('\n')
    print("PAGE CSV NUMBER: ",str(page))
    with open(current_dir+'/output/exported'+str(page)+'.csv', 'w', newline='') as csv_file:
        print(csv_file)
        csv_writer = csv.writer(csv_file)

        # write each lines
        for line in lines:
            # line = remove_foreign_characters(line)
            # skip for newline
            if line.strip() != '':
                # print('line split: ',line.split(','))
                csv_writer.writerow([remove_foreign_characters(item) for item in line.split(',')])

def normalize_string(input_string):
    # remove non-alphanumeric symbols/characters
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', input_string)
    # convert to lowercase
    normalized_string = cleaned_string.lower()
    return normalized_string


def remove_foreign_characters(input_string):
    # defining a regex pattern to match non-alphanumeric characters, and removing them
    pattern = re.compile(r'[^\w\s()]/')
    clean_string = re.sub(pattern, '', input_string)
    return clean_string

def extract_numbers_from_string(s):

    # regex to find all numbers in the string
    numbers = re.findall(r'\d+', s)

    # return the number (for match)
    return ', '.join(map(str, [int(num) for num in numbers]))
                     
def count_exported_csv_files(folder_path):
    files = os.listdir(folder_path)
    # counting how many exported<number>.csv files were made
    count = sum(1 for file in files if file.startswith("exported") and file.endswith(".csv"))
    return count

def delete_exported_csv_files(folder_path):
    files = os.listdir(folder_path)
    
    # delete exported csv files for a cleaner directory after sending it to the excel file.
    for file in files:
        if file.startswith("exported") and file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print("Deleted file:", file_path)

def delete_temp_files():
    folder_path = os.getcwd()
    files = os.listdir(folder_path)
    for file in files:
        if file.startswith("temp_page") and file.endswith(".png"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print("Deleted file:", file_path)

def find_index_of_2021_or_2022(data):
    # regex for matching years from 1900 to 2099 (to recognize where the year is indexed)
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')  
    list_of_years = []
    list_of_index = []

    # add to list_of_years and list_of_index if year and index matches to the regex
    [(list_of_years.append(match.group()), list_of_index.append(index)) for index, sublist in enumerate(data) for item in sublist if (match := year_pattern.search(item))]

    # make it into a list
    list_of_years = set(list_of_years)
    print('list of years :',list_of_years)
    return [list_of_index[0], list_of_years]


def find_columns_with_years(ws, years):
    column_indexes = []
    
    for cell in ws[1]:
        if str(cell.value) in years:
            
            column_indexes.append(cell.column)

    return column_indexes


def create_dictionary_from_index(data, start_index):
    result_dict = {}
    for sublist in data[start_index+1:]:
        result_dict[sublist[0]] = sublist[1:]
    return result_dict

# def statement_to_xlsx(input_statement,excel_file,dictionary):
#     cleaned_statement = input_statement.replace('|', ',').replace('/n', '\n')
#     lines = cleaned_statement.split('\n')


#     # creating a new workbook to make a worksheet
#     wb = Workbook()
#     ws = wb.active

#     data = []

#     for line in lines:
    
#         if line.strip() != '':
#             row_data = line.split(',')
#             data.append(row_data)
#             ws.append(row_data)

#     # saving the workbook
#     xlsx_file_path = os.path.join(os.getcwd(), 'output', 'exported.xlsx')
#     print('XLSX FILE PATH: ',xlsx_file_path)
#     wb.save(xlsx_file_path)

#     sheet_name = 'Financial Analysis'
#     column = 'D'

#     # finding the index of the sublist containing '2021' or '2022'
#     index_of_2021_or_2022,year = find_index_of_2021_or_2022(data)
#     if index_of_2021_or_2022 is not None:
#         print("Index of sublist containing '2021' or '2022':", index_of_2021_or_2022)
#     else:
#         print("No sublist containing '2021' or '2022' found.")

#     # creating a dictionary starting from the index where '2021' or '2022' is found
#     if index_of_2021_or_2022 is not None:
#         result_dict = create_dictionary_from_index(data, index_of_2021_or_2022)
        
#     else:
#         print("No sublist containing '2021' or '2022' found.")
    
#     write_to_excel(sheet_name, year,result_dict,excel_file,dictionary)
    


# def write_to_excel(sheet_name,year,result_dict,excel_file,dictionary):

    
#     wb = load_workbook(current_dir+'/'+excel_file)
#     print("WB: ",wb)
#     ws = wb[sheet_name]


#     result_dict_lower = {key.lower(): value for key, value in result_dict.items()}
#     print("result dict lower: ",result_dict_lower)
#     master_dictionary = {key.lower(): value for key,value in dictionary.items()}

#     print('master dictionary: ',master_dictionary)
#     indexes = [-2,-1]

#     # for result dictionary keys
#     n_result_dictionary_keys = {}
#     for key, value in result_dict.items():
#         normalized_key = normalize_string(key)
#         n_result_dictionary_keys[normalized_key] = value

#     # for master dictionary (items and key)
#     n_master_dictionary_keys = {}
#     for key, value in master_dictionary.items():
#         # Normalize the key
#         normalized_key = normalize_string(key)
        
#         # Normalize strings within lists
#         if isinstance(value, list):
#             normalized_values = [normalize_string(item) for item in value]
#             n_master_dictionary_keys[normalized_key] = normalized_values
#         else:
#             n_master_dictionary_keys[normalized_key] = normalize_string(value)

#     all_years_dictionaries = []
#     # Create sets of keys for quicker lookup
#     set_extracted_csv_keys = set(n_result_dictionary_keys.keys())

#     for kiof in indexes:
#         final_dictionary = {}
#         for current_master_key, current_master_list in n_master_dictionary_keys.items():
#             if current_master_key in set_extracted_csv_keys:
#                 extracted_csv_value = n_result_dictionary_keys[current_master_key][kiof]
#                 if extracted_csv_value not in [None, '']:
#                     final_dictionary[current_master_key] = extracted_csv_value

#             # checking for intersection between current master list and extracted CSV keys
#             common_keys = set(current_master_list) & set_extracted_csv_keys
#             if common_keys:
#                 index_of_alternate_term = current_master_list.index(next(iter(common_keys)))
#                 name_of_alt_term = current_master_list[index_of_alternate_term]
#                 final_dictionary[current_master_key] = n_result_dictionary_keys[name_of_alt_term][kiof]

#         all_years_dictionaries.append(final_dictionary)

#     list_of_year_indexes = find_columns_with_years(ws,year)
    
#     normalized_cell_values = {(normalize_string(cell[0].lower()), row_idx) for row_idx, cell in enumerate(ws.iter_rows(min_col=2, max_col=2, values_only=True), start=1) if cell[0]}

#     # Iterate over each year
#     for y, current_dictionary in enumerate(all_years_dictionaries):
#         for key, value in current_dictionary.items():
#             # Normalize the key
#             normalized_key = normalize_string(key)
#             # Check if the normalized key exists in the precomputed set
#             for normalized_cell_value, row_index in normalized_cell_values:
#                 if normalized_cell_value == normalized_key:
#                     col_letter = get_column_letter(list_of_year_indexes[y])
#                     cell_address = f"{col_letter}{row_index}"
#                     ws[cell_address] = value
#                     break
                
#     wb.save(current_dir+'/'+excel_file)
#     file_path = os.path.join(current_dir,excel_file)
#     os.system(f'open {file_path}') 

def statement_to_xlsx(input_statement,excel_file,dictionary):
    cleaned_statement = input_statement.replace('|', ',').replace('/n', '\n')
    lines = cleaned_statement.split('\n')


    # creating a new workbook to make a worksheet
    wb = Workbook()
    ws = wb.active

    data = []

    for line in lines:
    
        if line.strip() != '':
            row_data = line.split(',')
            data.append(row_data)
            ws.append(row_data)

    # saving the workbook
    xlsx_file_path = os.path.join(os.getcwd(), 'output', 'exported.xlsx')
    print('XLSX FILE PATH: ',xlsx_file_path)
    wb.save(xlsx_file_path)

    sheet_name = 'Financial Analysis'
    column = 'D'

    # finding the index of the sublist containing '2021' or '2022'
    index_of_2021_or_2022,year = find_index_of_2021_or_2022(data)
    if index_of_2021_or_2022 is not None:
        print("Index of sublist containing '2021' or '2022':", index_of_2021_or_2022)
    else:
        print("No sublist containing '2021' or '2022' found.")

    # creating a dictionary starting from the index where '2021' or '2022' is found
    if index_of_2021_or_2022 is not None:
        result_dict = create_dictionary_from_index(data, index_of_2021_or_2022)
        
    else:
        print("No sublist containing '2021' or '2022' found.")
    
    the_url = write_to_excel(sheet_name, year,result_dict,excel_file,dictionary)
    return the_url
    


def write_to_excel(sheet_name,year,result_dict,excel_file,dictionary):

    
    wb = load_workbook(current_dir+'/'+excel_file)
    print("WB: ",wb)
    ws = wb[sheet_name]


    result_dict_lower = {key.lower(): value for key, value in result_dict.items()}
    print("result dict lower: ",result_dict_lower)
    master_dictionary = {key.lower(): value for key,value in dictionary.items()}

    print('master dictionary: ',master_dictionary)
    indexes = [-2,-1]

    # for result dictionary keys
    n_result_dictionary_keys = {}
    for key, value in result_dict.items():
        normalized_key = normalize_string(key)
        n_result_dictionary_keys[normalized_key] = value

    # for master dictionary (items and key)
    n_master_dictionary_keys = {}
    for key, value in master_dictionary.items():
        # Normalize the key
        normalized_key = normalize_string(key)
        
        # Normalize strings within lists
        if isinstance(value, list):
            normalized_values = [normalize_string(item) for item in value]
            n_master_dictionary_keys[normalized_key] = normalized_values
        else:
            n_master_dictionary_keys[normalized_key] = normalize_string(value)

    all_years_dictionaries = []
    # Create sets of keys for quicker lookup
    set_extracted_csv_keys = set(n_result_dictionary_keys.keys())

    for kiof in indexes:
        final_dictionary = {}
        for current_master_key, current_master_list in n_master_dictionary_keys.items():
            if current_master_key in set_extracted_csv_keys:
                extracted_csv_value = n_result_dictionary_keys[current_master_key][kiof]
                if extracted_csv_value not in [None, '']:
                    final_dictionary[current_master_key] = extracted_csv_value

            # checking for intersection between current master list and extracted CSV keys
            common_keys = set(current_master_list) & set_extracted_csv_keys
            if common_keys:
                index_of_alternate_term = current_master_list.index(next(iter(common_keys)))
                name_of_alt_term = current_master_list[index_of_alternate_term]
                final_dictionary[current_master_key] = n_result_dictionary_keys[name_of_alt_term][kiof]

        all_years_dictionaries.append(final_dictionary)

    list_of_year_indexes = find_columns_with_years(ws,year)
    
    normalized_cell_values = {(normalize_string(cell[0].lower()), row_idx) for row_idx, cell in enumerate(ws.iter_rows(min_col=2, max_col=2, values_only=True), start=1) if cell[0]}

    # Iterate over each year
    for y, current_dictionary in enumerate(all_years_dictionaries):
        for key, value in current_dictionary.items():
            # Normalize the key
            normalized_key = normalize_string(key)
            # Check if the normalized key exists in the precomputed set
            for normalized_cell_value, row_index in normalized_cell_values:
                if normalized_cell_value == normalized_key:
                    col_letter = get_column_letter(list_of_year_indexes[y])
                    cell_address = f"{col_letter}{row_index}"
                    ws[cell_address] = value
                    break
    
    wb.save(current_dir+'/'+excel_file)
    # download_link = url_for('download_file', filename=excel_file)
    print('excel_file: ',excel_file)
    # file_name = f'exported.xlsx'
    file_name = os.path.basename(excel_file)
    download_link = '/download/' + file_name
    # Generate a download link for the temporary file
    # download_link = f'/download/{excel_file}'
    # print('download link: ',download_link)
    return download_link
# download_link = url_for('download_file', filename=excel_file)
    # print('excel_file: ',excel_file)
    # Generate a download link for the temporary file
    # download_link = f'/download/{excel_file}'
    # print('download link: ',download_link)
    # file_name = f'exported.xlsx'
    # file_name = os.path.basename(excel_file)
# just to try it on
# def main():

#     with open(current_dir+'/output/exported0.csv', 'r+') as file:
#         csv_data = file.read()
#         statement_to_xlsx(csv_data)

# if __name__ == "__main__":
#     main()