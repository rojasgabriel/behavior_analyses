#!/usr/bin/env python3
"""
Quick launcher for the Performance Dashboard.
Usage: python launch_dashboard.py
"""

import sys
from pathlib import Path

# Add the labdata2_testing directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard import app

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Performance Dashboard...")
    print("=" * 60)
    print("\nDashboard will be available at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server\n")
    print("=" * 60)
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
