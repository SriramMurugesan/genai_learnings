#!/bin/bash

echo "=========================================="
echo "Generative AI & Deep Learning Foundations"
echo "Environment Setup Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
 echo " Python 3 is not installed. Please install Python 3.8 or higher."
 exit 1
fi

echo " Python found: $(python3 --version)"
echo ""

# Create virtual environment
echo " Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo " Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo " Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo " source venv/bin/activate"
echo ""
echo "To download datasets, run:"
echo " python datasets/download_datasets.py"
echo ""
echo "To start Jupyter Lab, run:"
echo " jupyter lab"
echo ""
echo "Happy Learning! "
