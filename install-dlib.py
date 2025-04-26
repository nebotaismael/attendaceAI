#!/usr/bin/env python3
import subprocess
import sys
import platform

def install_dlib():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Try multiple installation methods
    methods = [
        # Method 1: Clone from GitHub and build
        "git clone https://github.com/davisking/dlib.git && cd dlib && python setup.py install",
        
        # Method 2: Using pip with minimal build
        [sys.executable, "-m", "pip", "install", "dlib", "--no-cache-dir", "--install-option=--no"],
        
        # Method 3: Using easy_install as fallback
        "easy_install dlib"
    ]
    
    for i, method in enumerate(methods):
        try:
            print(f"\nTrying installation method {i+1}...")
            if isinstance(method, list):
                subprocess.check_call(method)
            else:
                subprocess.check_call(method, shell=True)
            print(f"Success! dlib installed using method {i+1}")
            return True
        except Exception as e:
            print(f"Method {i+1} failed with error: {e}")
    
    print("All dlib installation methods failed.")
    return False

if __name__ == "__main__":
    install_dlib()