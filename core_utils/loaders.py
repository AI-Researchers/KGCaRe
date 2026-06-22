import logging
from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class HTMLDocsReader(BaseReader):
    """
    Custom HTML reader that parses HTML content into a structured list of Document objects.
    It tracks headers (like h1) and associates body text under the corresponding headings.
    """

    def __init__(self, *args: Any, remove_hyperlinks: bool = True, **kwargs: Any) -> None:
        self._remove_hyperlinks = remove_hyperlinks

    def html_to_docs(self, html_text: str, filename: str) -> List[Document]:
        """Convert raw HTML string to a list of Documents, grouped by header structure."""
        soup = BeautifulSoup(html_text, "html.parser")
        documents: List[Document] = []
        header_stack = []
        current_text = ""

        def get_full_header_path() -> str:
            return "/".join(header_stack)

        def save_current_text():
            nonlocal current_text
            if current_text.strip():
                documents.append(
                    Document(
                        text=current_text.strip(),
                        metadata={
                            "File Name": filename,
                            "Content Type": "text",
                            "Header Path": get_full_header_path(),
                        },
                    )
                )
                current_text = ""

        for element in soup.descendants:
            if element.name in ["h1"]:
                save_current_text()
                header_level = int(element.name[1])
                header_text = element.get_text().strip()

                while len(header_stack) >= header_level:
                    header_stack.pop()
                header_stack.append(header_text)

            elif element.name == "p":
                current_text += element.get_text() + "\n"

            elif element.name == "li":
                current_text += "- " + element.get_text() + "\n"

            elif element.name == "tr":
                cells = element.find_all("td")
                if cells:
                    current_text += " | ".join(cell.get_text() for cell in cells) + "\n"

        save_current_text()
        return documents

    def remove_hyperlinks(self, content: str) -> str:
        """Strip anchor tags (<a>) while preserving link text."""
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a"):
            a.unwrap()
        return str(soup)

    def parse_tups(self, filepath: Path) -> List[Document]:
        """Open and parse an HTML file into Documents."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)

        return self.html_to_docs(content, str(filepath))

    def load_data(self, file: Path, extra_info: Optional[Dict] = None) -> List[Document]:
        """Parse and enrich Document objects with optional metadata."""
        documents = self.parse_tups(file)

        # Inject additional metadata if provided
        for doc in documents:
            doc.metadata.update(extra_info or {})

        return documents


def load_html_docs(filepath: str, reader_type: str = "custom"):
    """
    Load HTML or plain text documents.
    
    Args:
        filepath (str): Path to the document directory.
        reader_type (str): "custom" (for ConditionalQA) or "default" (for HotpotQA).
        
    Returns:
        nodes (List[TextNode]): Loaded documents as nodes.
    """
    logger.info(f"Loading documents from {filepath} using reader: {reader_type}...")

    if reader_type == "custom":
        file_extractor = {".txt": HTMLDocsReader(tags=["h1"])}
    else:
        file_extractor = None  # Let LlamaIndex decide

    loader = SimpleDirectoryReader(
        input_dir=filepath,
        exclude=[
            "*.rst", "*.ipynb", "*.py", "*.bat", "*.png", "*.jpg", "*.jpeg", "*.csv",
            "*.html", "*.js", "*.css", "*.pdf", "*.json"
        ],
        file_extractor=file_extractor,
        recursive=True
    )

    nodes = loader.load_data()
    logger.info("Documents loaded successfully.")
    return nodes


def modify_metadata(nodes):
    """
    Add consistent metadata format to document nodes.
    """
    logger.info("Modifying metadata for nodes...")

    text_template = "Content Metadata:\n{metadata_str}\n\nContent:\n{content}"
    metadata_template = "{key}: {value},"

    for doc in nodes:
        doc.text_template = text_template
        doc.metadata_template = metadata_template
        doc.excluded_llm_metadata_keys = [
            "File Name", "file_type", "file_size", "creation_date",
            "last_modified_date", "last_accessed_date", "file_path", "Content Type"
        ]
        doc.excluded_embed_metadata_keys = doc.excluded_llm_metadata_keys

    logger.info("Metadata modification completed.")
    return nodes