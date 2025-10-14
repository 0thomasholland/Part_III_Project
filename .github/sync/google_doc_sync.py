import json
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configuration
SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
DOC_ID = os.environ['GOOGLE_DOC_ID']
OUTPUT_FILE = 'work_log/Progress.md'  # Change to your target file

def get_credentials():
    """Load credentials from environment variable."""
    creds_json = os.environ['GOOGLE_API_JSON']
    creds_dict = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES)
    return credentials

def extract_text_from_doc(document):
    """Convert Google Doc to markdown-style text."""
    content = []
    
    for element in document.get('body', {}).get('content', []):
        if 'paragraph' in element:
            paragraph = element['paragraph']
            para_text = []
            
            for text_run in paragraph.get('elements', []):
                if 'textRun' in text_run:
                    text_content = text_run['textRun'].get('content', '')
                    para_text.append(text_content)
            
            full_text = ''.join(para_text)
            
            # Handle paragraph styles
            style = paragraph.get('paragraphStyle', {})
            named_style = style.get('namedStyleType', '')
            
            if named_style == 'HEADING_1':
                content.append(f"# {full_text}")
            elif named_style == 'HEADING_2':
                content.append(f"## {full_text}")
            elif named_style == 'HEADING_3':
                content.append(f"### {full_text}")
            elif named_style == 'HEADING_4':
                content.append(f"#### {full_text}")
            elif full_text.strip():
                content.append(full_text)
    
    return ''.join(content)

def main():
    # Authenticate and build service
    credentials = get_credentials()
    service = build('docs', 'v1', credentials=credentials)
    
    # Fetch the document
    document = service.documents().get(documentId=DOC_ID).execute()
    
    # Convert to markdown
    markdown_content = extract_text_from_doc(document)
    
    # Write to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Successfully updated {OUTPUT_FILE}")

if __name__ == '__main__':
    main()