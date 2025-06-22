#!/usr/bin/env python3
"""
Server Status Checker
=====================
Checks if both Streamlit and GitHub Pages are accessible
"""

import requests
import socket
import time

def check_local_streamlit():
    """Check if local Streamlit server is running"""
    try:
        # Check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 8501))
        sock.close()
        
        if result == 0:
            print("✅ Streamlit server is running on http://localhost:8501")
            return True
        else:
            print("❌ Streamlit server is not accessible")
            return False
    except Exception as e:
        print(f"❌ Error checking Streamlit: {e}")
        return False

def check_github_pages():
    """Check if GitHub Pages is accessible"""
    try:
        url = "https://gauravpatil2515.github.io/skin-cancer-classification/"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ GitHub Pages is live at https://gauravpatil2515.github.io/skin-cancer-classification/")
            return True
        else:
            print(f"❌ GitHub Pages returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking GitHub Pages: {e}")
        return False

def main():
    """Check both servers"""
    print("🔍 Checking server status...\n")
    
    streamlit_ok = check_local_streamlit()
    github_ok = check_github_pages()
    
    print(f"\n📊 Status Summary:")
    print(f"   Local Streamlit: {'✅ Running' if streamlit_ok else '❌ Down'}")
    print(f"   GitHub Pages: {'✅ Live' if github_ok else '❌ Down'}")
    
    if streamlit_ok and github_ok:
        print("\n🎉 Both servers are operational!")
        print("\n🚀 Access your applications:")
        print("   Local App: http://localhost:8501")
        print("   GitHub Pages: https://gauravpatil2515.github.io/skin-cancer-classification/")
    else:
        print("\n⚠️ Some servers need attention.")

if __name__ == "__main__":
    main()
