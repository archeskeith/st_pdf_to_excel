#!/bin/bash

# Deactivate the existing virtualenv
if [ -f venv/bin/deactivate ]; then
    echo "Deactivating existing virtualenv..."
    source venv/bin/deactivate
fi

# Create and activate a new virtualenv
echo "Creating and activating a new virtualenv..."
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

