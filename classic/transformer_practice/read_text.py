import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def extract_text(book_path):
    """Extract plain text from EPUB file"""
    book = epub.read_epub(book_path)

    text_content = []
    # Iterate through all chapters in the book
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), "html.parser")
            # Extract plain text
            text = soup.get_text()
            if text.strip():  # If content is not empty
                text_content.append(text)

    return "\n".join(text_content)


def clean_text(text):
    """Simple text cleaning"""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines and lines that are too short
        if len(line) > 5:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


if __name__ == "__main__":
    book_path = "data/三國演義.epub"
    text = extract_text(book_path)
    cleaned_text = clean_text(text)
    print(cleaned_text)
