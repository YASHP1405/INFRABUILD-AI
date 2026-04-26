"""
Wrapper script to generate PDF from pythonTXT.txt content.
Adjusts the output path to the user's Downloads folder on Windows.
"""
import sys
import os

# Add the source file's directory for any relative imports if needed
sys.path.insert(0, r"c:\Users\Yash\Downloads")

# Read and execute the source script with modified output path
with open(r"c:\Users\Yash\Downloads\pythonTXT.txt", "r", encoding="utf-8") as f:
    source_code = f.read()

# The file starts with "python\n" (a language marker), strip it
if source_code.startswith("python\n"):
    source_code = source_code[len("python\n"):]
elif source_code.startswith("python\r\n"):
    source_code = source_code[len("python\r\n"):]

# Replace the Linux output path with a Windows path in Downloads
source_code = source_code.replace(
    '/mnt/user-data/outputs/Economics_Answer_Book_Summer2026.pdf',
    r'c:/Users/Yash/Downloads/Economics_Answer_Book_Summer2026.pdf'
)

exec(source_code)
