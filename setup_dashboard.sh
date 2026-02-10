#!/bin/bash
# Setup script for the Performance Dashboard
# Run this script to install all required dependencies

echo "=================================================="
echo "Performance Dashboard - Setup Script"
echo "=================================================="
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null
then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

echo "Installing dashboard dependencies..."
pip install -r labdata2_testing/dashboard_requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Installation complete!"
    echo "=================================================="
    echo ""
    echo "To launch the dashboard, run:"
    echo "  cd labdata2_testing"
    echo "  python launch_dashboard.py"
    echo ""
    echo "The dashboard will be available at:"
    echo "  http://127.0.0.1:8050"
    echo ""
else
    echo ""
    echo "✗ Installation failed. Please check the error messages above."
    exit 1
fi
