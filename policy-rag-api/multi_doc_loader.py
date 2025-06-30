import os
import re
import glob
import yaml
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def parse_frontmatter(md_text):
    """
    Extract YAML frontmatter (between ---) and return metadata dict + content.
    """
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", md_text, re.DOTALL)
    if match:
        meta_raw, content = match.groups()
        metadata = yaml.safe_load(meta_raw)
        return metadata, content
    else:
        # No frontmatter found
        return {}, md_text

def load_and_chunk_markdown_docs(folder, chunk_size=1000) -> List[Document]:
    """
    Loads all .md files in the folder, extracts metadata, chunks content
    using MarkdownHeaderTextSplitter (header 1-4), and returns a list of Document objects.
    Each chunk contains the document-level metadata + header section info.
    Preserves tables and code blocks as much as possible.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        separators=["\n\n", "\n", "|", " ", ""],
        keep_separator=True
    )
    docs = []
    for md_file in glob.glob(os.path.join(folder, "*.md")):
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        metadata, content = parse_frontmatter(md_text)
        # Use MarkdownHeaderTextSplitter to keep header structure
        header_chunks = markdown_splitter.split_text(content)
        for doc in header_chunks:
            # Further split if too large (while preserving metadata)
            if len(doc.page_content) > chunk_size:
                sub_docs = text_splitter.split_documents([doc])
                for sub_doc in sub_docs:
                    # Merge doc-level metadata and header metadata
                    sub_doc.metadata = {**metadata, **sub_doc.metadata}
                    docs.append(sub_doc)
            else:
                doc.metadata = {**metadata, **doc.metadata}
                docs.append(doc)
    return docs

# --- Example usage ---
if __name__ == "__main__":
    folder = "./knowledgebase"
    chunk_size = 1000
    all_chunks = load_and_chunk_markdown_docs(folder, chunk_size=chunk_size)
    print(f"Loaded {len(all_chunks)} chunks from {folder}")
    # Each chunk is a langchain.schema.Document with page_content and metadata
    for i, doc in enumerate(all_chunks[:3]):
        print(f"Chunk {i+1}:")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:120], "...")
        print("-" * 40)