import streamlit as st


# Function to set favicon
def set_favicon(favicon_path):
    with open(favicon_path, "rb") as f:
        favicon_bytes = f.read()
        st.set_page_config(page_icon=favicon_bytes, layout="wide")


def failed_page():
    st.markdown(
        """
        ## Failed
        Your upload failed. Please try again.
        """,
        unsafe_allow_html=True,
    )