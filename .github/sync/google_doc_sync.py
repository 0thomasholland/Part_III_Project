import json
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build

# Configuration
SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
DOC_ID = os.environ['GOOGLE_DOC_ID']
OUTPUT_FILE = "work_log/Progress.md"

def get_credentials():
    """Load credentials from environment variable."""
    creds_json = os.environ['GOOGLE_API_JSON']
    creds_dict = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES)
    return credentials

def get_list_item_prefix(paragraph):
    """Extract list formatting (bullet or number) from paragraph."""
    bullet = paragraph.get("bullet")
    if not bullet:
        return None

    # Get nesting level
    nesting_level = bullet.get("nestingLevel", 0)
    indent = "  " * nesting_level

    # Check if it's a numbered list
    list_id = bullet.get("listId")
    if list_id:
        # Numbered lists have glyph format
        glyph_format = bullet.get("glyphFormat", "")
        if glyph_format and "%" in glyph_format:
            return f"{indent}1. "

    # Default to bullet point
    return f"{indent}- "


def extract_text_from_doc(document):
    """Convert Google Doc to markdown-style text."""
    content = []
    
    for element in document.get('body', {}).get('content', []):
        if 'paragraph' in element:
            paragraph = element['paragraph']
            para_text = []

            # Extract text content
            for text_run in paragraph.get('elements', []):
                if 'textRun' in text_run:
                    text_content = text_run['textRun'].get('content', '')
                    para_text.append(text_content)
            
            full_text = ''.join(para_text)

            # Skip empty paragraphs (but preserve intentional line breaks)
            if not full_text.strip():
                content.append("\n")
                continue

            # Handle list items
            list_prefix = get_list_item_prefix(paragraph)
            if list_prefix:
                # Remove trailing newline from text, add it back after prefix
                text = full_text.rstrip("\n")
                content.append(f"{list_prefix}{text}\n")
                continue

            # Handle paragraph styles (headings)
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
            else:
                # Regular paragraph - preserve as is
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