"""
Test script to examine UnstructuredLoader element metadata
"""
from langchain_unstructured import UnstructuredLoader
import json

loader = UnstructuredLoader('tests/text_pdfs/pmc_32548317.pdf', mode='elements')
docs = loader.load()

print(f"Total elements: {len(docs)}")
print("\nFirst 20 elements with metadata:")
print("-" * 80)

for i, doc in enumerate(docs[:20]):
    category = doc.metadata.get('category', 'unknown')
    content = doc.page_content[:60].replace('\n', ' ')
    print(f"{i:3d} | {category:20s} | {content}...")

print("\n\nUnique categories found:")
categories = set(doc.metadata.get('category', 'unknown') for doc in docs)
for cat in sorted(categories):
    count = sum(1 for d in docs if d.metadata.get('category') == cat)
    print(f"  {cat}: {count}")
