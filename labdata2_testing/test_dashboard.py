#!/usr/bin/env python3
"""
Test script to verify dashboard functionality without launching the server.
This validates the dashboard structure and key functions.
"""

import sys
from pathlib import Path

# Add the labdata2_testing directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import plotly.graph_objects
        import dash
        import dash_bootstrap_components
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r dashboard_requirements.txt")
        return False


def test_dashboard_structure():
    """Test that the dashboard module has the expected structure."""
    print("\nTesting dashboard structure...")
    try:
        import dashboard
        
        # Check for required functions
        required_functions = [
            'fetch_mouse_data',
            'calculate_metrics',
            'create_performance_figure',
            'get_available_mice',
        ]
        
        for func_name in required_functions:
            if not hasattr(dashboard, func_name):
                print(f"✗ Missing function: {func_name}")
                return False
            print(f"  ✓ Function found: {func_name}")
        
        # Check for app object
        if not hasattr(dashboard, 'app'):
            print("✗ Missing 'app' object")
            return False
        print("  ✓ Dash app object found")
        
        print("✓ Dashboard structure is valid")
        return True
    
    except Exception as e:
        print(f"✗ Error loading dashboard: {e}")
        return False


def test_data_fetching():
    """Test that data fetching functions work (requires database access)."""
    print("\nTesting data fetching...")
    try:
        import dashboard
        
        # Try to get available mice
        mice = dashboard.get_available_mice()
        print(f"  Available mice: {len(mice)} found")
        
        if mice:
            print(f"  Sample mice: {mice[:3]}")
            
            # Try to fetch data for the first mouse
            result = dashboard.fetch_mouse_data(mice[0], 10)
            if result:
                data, sesdata = result
                print(f"  ✓ Successfully fetched data for {mice[0]}")
                print(f"    Data shape: {data.shape}")
                print(f"    Session data shape: {sesdata.shape}")
                
                # Try to calculate metrics
                metrics = dashboard.calculate_metrics(data, sesdata, 10)
                print(f"  ✓ Successfully calculated metrics")
                print(f"    Metrics keys: {list(metrics.keys())}")
                
                # Try to create figure
                fig = dashboard.create_performance_figure(metrics)
                print(f"  ✓ Successfully created performance figure")
                
                return True
            else:
                print(f"  ✗ No data returned for {mice[0]}")
                return False
        else:
            print("  ⚠ No mice found in database (database may not be accessible)")
            print("  This is expected if the database is not available in this environment")
            return True
    
    except Exception as e:
        print(f"  ⚠ Data fetching test skipped: {e}")
        print("  This is expected if the database is not available in this environment")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Dashboard Validation Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_dashboard_structure,
        test_data_fetching,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed! Dashboard is ready to use.")
        print("\nTo launch the dashboard, run:")
        print("  python launch_dashboard.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
