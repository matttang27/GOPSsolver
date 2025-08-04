#!/usr/bin/env python3
"""
Type checking script for GOPS solver.
Runs mypy static type analysis.
"""

import subprocess
import sys
import os

def install_mypy():
    """Install mypy if not already installed."""
    try:
        import mypy
        print("✅ mypy is already installed")
        return True
    except ImportError:
        print("📦 Installing mypy...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "mypy"], 
                         check=True, capture_output=True)
            print("✅ mypy installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install mypy: {e}")
            return False

def run_type_check():
    """Run mypy type checking on all Python files."""
    if not install_mypy():
        return False
    
    print("\n🔍 Running type analysis with mypy...")
    print("="*50)
    
    try:
        # Run mypy on the main files
        result = subprocess.run([
            sys.executable, "-m", "mypy", 
            "example.py", "test_example.py"
        ], capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print("🎉 No type errors found!")
            return True
        else:
            print(f"❌ Type checking failed with return code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running mypy: {e}")
        return False
    except FileNotFoundError:
        print("❌ mypy not found. Please install it with: pip install mypy")
        return False

def check_runtime_types():
    """Demonstrate runtime type checking with typeguard."""
    print("\n🛡️  Runtime Type Checking Example")
    print("="*50)
    
    try:
        # Try to import typeguard for runtime checking
        import typeguard
        print("✅ typeguard is available for runtime type checking")
        
        # Example of how to use it
        print("\nTo enable runtime type checking, decorate functions with:")
        print("@typeguard.typechecked")
        print("def your_function(param: int) -> str:")
        print("    return str(param)")
        
    except ImportError:
        print("📦 Install typeguard for runtime type checking:")
        print("pip install typeguard")

if __name__ == "__main__":
    print("🐍 Python Type Checking Tool")
    print("="*50)
    
    success = run_type_check()
    check_runtime_types()
    
    if success:
        print("\n✅ Type checking completed successfully!")
    else:
        print("\n❌ Type checking found issues. Please fix them.")
    
    sys.exit(0 if success else 1)
