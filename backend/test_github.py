import json
import base64
from github_analyzer import fetch_repo_data, generate_repo_questions

repo_data = fetch_repo_data("torvalds")

questions = generate_repo_questions(repo_data)
print("Questions generated:")
for q in questions:
    print(q)
