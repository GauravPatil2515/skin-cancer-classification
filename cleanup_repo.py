#!/usr/bin/env python3
"""
Repository Cleanup Script
========================
Cleans up the repository by removing unnecessary files and keeping only what's needed.
"""

import os
import shutil
import sys

def clean_repository():
    """Remove unnecessary files and keep only what's needed for the project."""
    
    # Files to keep
    essential_files = [
        'main_app.py',
        'requirements.txt',
        'README.md',
        'LICENSE',
        'PROJECT_STRUCTURE.md',
        'index.html',
        'documentation.html',
        '404.html',
        '_config.yml',
        '.gitignore',
    ]
    
    # Directories to keep
    essential_dirs = [
        'src',
        'assets',
        'config',
        'docs',
        '.github',
    ]
    
    # Get list of all files and directories in the current directory
    all_items = os.listdir('.')
    
    # Remove files that are not in the essential list
    for item in all_items:
        # Skip if it's a directory or an essential file
        if os.path.isdir(item) and item not in essential_dirs:
            if item != '.git':  # Don't delete the .git directory
                print(f"Removing directory: {item}")
                shutil.rmtree(item, ignore_errors=True)
        elif os.path.isfile(item) and item not in essential_files:
            if not item.startswith('.') and item != 'cleanup_repo.py':  # Skip hidden files and this script
                print(f"Removing file: {item}")
                os.remove(item)
    
    print("\n✅ Repository cleanup complete!")

if __name__ == "__main__":
    print("🧹 Starting repository cleanup...")
    clean_repository()
