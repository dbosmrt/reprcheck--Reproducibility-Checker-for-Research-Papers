"""
PDF Parser - Using UnstructuredLoader Element Metadata
Properly extracts sections using element categories and smart grouping.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_unstructured import UnstructuredLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a paper section with title and content."""
    name: str
    title: str = ""
    content: str = ""
    elements: List[str] = field(default_factory=list)


class FileLoader:
    """Load PDF files from a folder."""
    
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")
        logger.info(f"Folder validated: {self.folder_path}")
    
    def load(self) -> List[Path]:
        pdf_files = list(self.folder_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files


class PDFProcessor:
    """Load PDF and extract elements with metadata."""
    
    # Section keywords mapping
    SECTION_KEYWORDS = {
        "abstract": ["abstract", "summary", "a b s t r a c t"],
        "introduction": ["introduction", "1. introduction", "1 introduction", "background"],
        "materials": ["materials", "materials and methods", "material and method", "data", "dataset"],
        "methodology": ["method", "methods", "methodology", "approach", "proposed method", 
                       "experimental setup", "model", "framework", "procedure", "techniques"],
        "results": ["result", "results", "experiments", "findings", "experimental results", 
                   "results and discussion", "outcomes", "analysis"],
        "discussion": ["discussion", "analysis", "interpretation"],
        "conclusion": ["conclusion", "conclusions", "concluding remarks", "summary and conclusion"],
        "references": ["references", "bibliography", "literature cited", "works cited", "reference"],
        "acknowledgements": ["acknowledgement", "acknowledgements", "acknowledgment", "funding"],
    }
    
    def __init__(self, mode: str = "elements", strategy: str = "fast"):
        self.mode = mode
        self.strategy = strategy
    
    def load(self, file_path: str) -> Tuple[List[Any], str]:
        """Load PDF and return elements with full text."""
        try:
            loader = UnstructuredLoader(str(file_path), mode=self.mode, strategy=self.strategy)
            documents = loader.load()
            full_text = "\n\n".join([doc.page_content for doc in documents])
            logger.info(f"Loaded {len(documents)} elements from {Path(file_path).name}")
            return documents, full_text
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def extract_structured_data(self, documents: List[Any], full_text: str, file_name: str) -> Dict[str, Any]:
        """Extract structured data from document elements."""
        logger.info(f"Extracting structured data from {file_name}")
        
        # Initialize result structure
        result = {
            "file_name": file_name,
            "title": "",
            "authors": [],
            "abstract": "",
            "keywords": [],
            "sections": {},
            "references": [],
            "full_text": full_text,
        }
        
        # Group elements by category
        elements_by_category = self._group_by_category(documents)
        
        # Extract title (first Title element that looks like a paper title)
        result["title"] = self._extract_title(documents)
        
        # Extract authors
        result["authors"] = self._extract_authors(documents)
        
        # Extract keywords
        result["keywords"] = self._extract_keywords(documents)
        
        # Find and extract sections
        sections = self._extract_sections(documents)
        
        # Map sections to standard names
        for section_name in ["abstract", "introduction", "materials", "methodology", 
                            "results", "discussion", "conclusion", "references"]:
            result[section_name] = sections.get(section_name, "")
        
        # Parse references into list
        if result["references"]:
            result["references"] = self._parse_references(result["references"])
        
        # Count extracted sections
        extracted_count = sum(1 for k in ["abstract", "introduction", "materials", "methodology", 
                                          "results", "discussion", "conclusion"] 
                             if result.get(k))
        logger.info(f"Extracted {extracted_count} sections from {file_name}")
        
        return result
    
    def _group_by_category(self, documents: List[Any]) -> Dict[str, List[Any]]:
        """Group elements by their category."""
        groups = {}
        for doc in documents:
            category = doc.metadata.get("category", "unknown")
            if category not in groups:
                groups[category] = []
            groups[category].append(doc)
        return groups
    
    def _extract_title(self, documents: List[Any]) -> str:
        """Extract paper title from elements."""
        # Look for the first substantial Title element
        for doc in documents[:20]:
            category = doc.metadata.get("category", "")
            content = doc.page_content.strip()
            
            # Skip short or boilerplate titles
            if category in ["Title", "UncategorizedText"]:
                if len(content) > 20 and len(content) < 300:
                    # Skip common non-title patterns
                    if not re.match(r'(?i)^(contents|journal|research article|arxiv|doi|http)', content):
                        if not '@' in content:  # Not an email/author line
                            return content
        return ""
    
    def _extract_authors(self, documents: List[Any]) -> List[str]:
        """Extract author names from document."""
        authors = []
        
        # Look in first 30 elements for author information
        for doc in documents[:30]:
            content = doc.page_content.strip()
            category = doc.metadata.get("category", "")
            
            # Skip if this looks like abstract or keywords
            if re.match(r'(?i)^(abstract|keyword|introduction)', content):
                break
            
            # Look for name patterns in Title and UncategorizedText
            if category in ["Title", "UncategorizedText"]:
                # Pattern: Multiple names separated by comma or "and"
                if ',' in content or ' and ' in content.lower():
                    # Extract names
                    names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)', content)
                    for name in names:
                        if 4 < len(name) < 40 and name not in authors:
                            # Validate it's likely a name
                            if not re.search(r'(University|Institute|Department|College)', name):
                                authors.append(name)
        
        return authors[:20]
    
    def _extract_keywords(self, documents: List[Any]) -> List[str]:
        """Extract keywords from document."""
        for doc in documents[:40]:
            content = doc.page_content.strip()
            
            if re.match(r'(?i)^keywords?\s*[:\-]?', content):
                # Extract everything after "Keywords:"
                kw_text = re.sub(r'(?i)^keywords?\s*[:\-]?\s*', '', content)
                keywords = re.split(r'[;,·•]', kw_text)
                return [kw.strip() for kw in keywords if 2 < len(kw.strip()) < 50][:15]
        
        return []
    
    def _extract_sections(self, documents: List[Any]) -> Dict[str, str]:
        """Extract all sections by finding section headers and content."""
        sections = {}
        current_section = None
        current_content = []
        
        for doc in documents:
            content = doc.page_content.strip()
            category = doc.metadata.get("category", "")
            
            # Check if this is a section header
            section_type = self._identify_section_header(content, category)
            
            if section_type:
                # Save previous section
                if current_section and current_content:
                    section_text = "\n\n".join(current_content)
                    # Keep the longer version if section already exists
                    if current_section not in sections or len(section_text) > len(sections[current_section]):
                        sections[current_section] = section_text
                
                # Start new section
                current_section = section_type
                current_content = []
            elif current_section:
                # Add content to current section
                if len(content) > 10:  # Skip very short elements
                    current_content.append(content)
        
        # Save last section
        if current_section and current_content:
            section_text = "\n\n".join(current_content)
            if current_section not in sections or len(section_text) > len(sections[current_section]):
                sections[current_section] = section_text
        
        return sections
    
    def _identify_section_header(self, content: str, category: str) -> Optional[str]:
        """Identify if content is a section header and return section type."""
        # Section headers are usually short
        if len(content) > 100:
            return None
        
        content_lower = content.lower().strip()
        # Remove leading numbers and punctuation
        content_clean = re.sub(r'^[\d\.\s]+', '', content_lower).strip()
        content_clean = re.sub(r'[:\-–—].*$', '', content_clean).strip()
        
        # Check if it's a header category or ListItem (often section headers)
        if category in ["Title", "Header", "ListItem", "UncategorizedText"]:
            for section_type, keywords in self.SECTION_KEYWORDS.items():
                for keyword in keywords:
                    if content_clean == keyword or content_clean.startswith(keyword + " "):
                        return section_type
        
        return None
    
    def _parse_references(self, refs_text: str) -> List[Dict[str, str]]:
        """Parse references text into structured list."""
        if not refs_text or len(refs_text) < 50:
            return []
        
        references = []
        
        # Try splitting by [1], [2] format
        bracket_parts = re.split(r'\[(\d+)\]', refs_text)
        if len(bracket_parts) > 3:
            for i in range(1, len(bracket_parts), 2):
                if i + 1 < len(bracket_parts):
                    ref_id = bracket_parts[i]
                    ref_text = bracket_parts[i + 1].strip()
                    if len(ref_text) > 20:
                        references.append({
                            "id": int(ref_id),
                            "text": ref_text[:500]
                        })
            if references:
                return references
        
        # Try splitting by numbered format: 1., 2., etc.
        numbered_parts = re.split(r'\n\s*(\d+)\.\s+', refs_text)
        if len(numbered_parts) > 3:
            for i in range(1, len(numbered_parts), 2):
                if i + 1 < len(numbered_parts):
                    ref_id = numbered_parts[i]
                    ref_text = numbered_parts[i + 1].strip()
                    if len(ref_text) > 20:
                        references.append({
                            "id": int(ref_id),
                            "text": ref_text[:500]
                        })
            if references:
                return references
        
        # Fallback: split by double newlines
        paragraphs = re.split(r'\n\n+', refs_text)
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) > 30:
                references.append({
                    "id": i + 1,
                    "text": para[:500]
                })
        
        return references


class JSONSaver:
    """Save parsed content as JSON files."""
    
    def __init__(self, output_dir: str = "data/jsons"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def save(self, data: Dict[str, Any], filename: Optional[str] = None,
             include_full_text: bool = False) -> str:
        """Save data as JSON file."""
        if filename is None:
            filename = data.get("file_name", "output")
        
        # Optionally exclude full_text
        save_data = data.copy()
        if not include_full_text:
            save_data.pop("full_text", None)
        
        json_file = self.output_dir / f"{Path(filename).stem}.json"
        
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved: {json_file.name}")
            return str(json_file)
        except Exception as e:
            logger.error(f"Failed to save {json_file}: {e}")
            raise


def process_pdf(pdf_path: str, output_dir: str = "data/jsons",
                include_full_text: bool = False) -> str:
    """Process a single PDF and save as structured JSON."""
    processor = PDFProcessor()
    documents, full_text = processor.load(pdf_path)
    
    result = processor.extract_structured_data(documents, full_text, Path(pdf_path).name)
    
    saver = JSONSaver(output_dir)
    return saver.save(result, include_full_text=include_full_text)


def process_folder(input_folder: str, output_folder: str = "data/jsons",
                   include_full_text: bool = False) -> List[str]:
    """Process all PDFs in a folder."""
    loader = FileLoader(input_folder)
    pdf_files = loader.load()
    
    if not pdf_files:
        logger.warning("No PDF files found!")
        return []
    
    results = []
    for pdf in pdf_files:
        try:
            json_path = process_pdf(str(pdf), output_folder, include_full_text)
            results.append(json_path)
        except Exception as e:
            logger.error(f"Failed: {pdf.name} - {e}")
    
    logger.info(f"Completed: {len(results)}/{len(pdf_files)} files")
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/jsons"
        include_full = "--full" in sys.argv
        
        if Path(input_path).is_file():
            process_pdf(input_path, output_dir, include_full)
        else:
            process_folder(input_path, output_dir, include_full)
    else:
        print("Usage: python parser.py <pdf_file_or_folder> [output_folder] [--full]")
