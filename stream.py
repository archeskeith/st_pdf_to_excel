import subprocess
import re
import streamlit as st
import atexit

def extract_link(text):
    # Define a regular expression pattern to match URLs
    pattern = r'https?://\S+'
    
    # Search for the URL in the text
    match = re.search(pattern, text)
    
    # If a match is found, return the URL
    if match:
        print('match found')
        print(match.group())
        return match.group()
    else:
        return None

# Check if Flask with Ngrok is already running
process = subprocess.run(['pgrep', '-f', 'flask run'], capture_output=True, text=True)

if process.stdout:
    print("Flask with Ngrok is already running.")
    # Extract the Ngrok URL from the process output
    flask_url = extract_link(process.stdout)
else:
    print("Starting Flask with Ngrok...")
    # Start Flask with Ngrok
    process = subprocess.Popen(['flask', 'run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Wait for Flask to start and extract the Ngrok URL
    flask_url = None
    for line in process.stdout:
        print(line)
        flask_url = extract_link(line)
        if flask_url:
            break

# If Ngrok URL is found, embed it in Streamlit
if flask_url:
    st.markdown(f'<div style="width: 100vw; height: 100vh;"><iframe src="{flask_url}" width="100%" height="100%" style="border: none;"></iframe></div>', unsafe_allow_html=True)
else:
    st.write("Failed to start Flask with Ngrok.")

# Function to stop Flask server and close Ngrok tunnel
def cleanup():
    process.kill()

# Register cleanup function to execute when the script exits
    atexit.register(cleanup)