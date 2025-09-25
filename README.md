# Streamlit PDF to Excel/CSV Converter 🚀

This is a web-based application built with Streamlit that automates the tedious process of extracting tabular data from PDF files and converting it into structured Excel or CSV formats. This tool was designed to solve a common pain point for finance and operations teams who frequently work with PDF reports.



### ▶️ Live Demo

You can try the live application here: [https://advance-pdf-to-excel.streamlit.app/]

---

### The Problem: Unstructured Data in PDFs

In many business workflows, critical data is locked away in PDF documents, such as bank statements, invoices, or financial reports. Manually copying and pasting this data into a spreadsheet is slow, prone to human error, and incredibly inefficient. For my finance colleagues, this process could take hours each month.

---

### The Solution: An Automated Web Tool

I developed this application to provide a simple, user-friendly solution. Users can simply upload a PDF file, and the application's backend—powered by Python—parses the document, intelligently identifies tables, and extracts the data into a clean, usable format.

**Key Features:**
* **File Upload:** Simple drag-and-drop interface for uploading PDF files.
* **Multi-Page Support:** Ability to specify which page of the PDF to extract data from.
* **Format Selection:** Users can download the extracted data as either an Excel (`.xlsx`) or a CSV (`.csv`) file.


### 🛠️ Tech Stack

* **Web Framework:** `Streamlit`
* **Data Processing:** `Pandas`
* **PDF Parsing:** `PyPDF2`, `camelot-py` (or other libraries you used)
* **Deployment:** `Streamlit Cloud`

---

### How to Run Locally

1.  Clone the repository:
    ```bash
    git clone [https://github.com/archeskeith/st_pdf_to_excel.git](https://github.com/archeskeith/st_pdf_to_excel.git)
    cd st_pdf_to_excel
    ```
2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
