"""
PDF Parser Module - Simple and Clean Implementation
Classes: FileLoader, PDFProcessor, ContentParser, JSONSaver
"""
import os
import json
import logging
import regex as re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

from langchain_unstructured import UnstructuredLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FILE LOADER - Handles different file types
# =============================================================================

class FileLoader:
    """Loads files based on their extension (.pdf, .txt, .docx, .xml)."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.xml'}
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self._validate()
    
    def _validate(self) -> None:
        """Validate file exists and has supported extension."""
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {self.file_path.suffix}")
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}. Supported: {self.SUPPORTED_EXTENSIONS}")
    
    @property
    def extension(self) -> str:
        """Get file extension."""
        return self.file_path.suffix.lower()
    
    def load(self) -> List[Any]:
        """Load file content based on extension."""
        logger.info(f"Loading file: {self.file_path.name} ({self.extension})")
        
        try:
            loader = UnstructuredLoader(str(self.file_path))
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} elements from {self.file_path.name}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load {self.file_path.name}: {e}")
            raise


# =============================================================================
# PDF PROCESSOR - Processes files one at a time
# =============================================================================

class PDFProcessor:
    """Processes PDF/document files one at a time or in parallel."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def process_single(self, file_path: str) -> Dict[str, Any]:
        """Process a single file and return raw content."""
        logger.info(f"Processing: {file_path}")
        
        loader = FileLoader(file_path)
        documents = loader.load()
        
        # Combine document content
        full_text = "\n\n".join([doc.page_content for doc in documents])
        
        return {
            "file_name": Path(file_path).name,
            "file_path": str(file_path),
            "extension": loader.extension,
            "content": full_text,
            "element_count": len(documents),
            "processed_at": datetime.now().isoformat()
        }
    
    def process_batch(self, file_paths: List[str], parallel: bool = True) -> List[Dict[str, Any]]:
        """Process multiple files (parallel or sequential)."""
        results = []
        
        if parallel and len(file_paths) > 1:
            logger.info(f"Processing {len(file_paths)} files in parallel (workers={self.max_workers})")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.process_single, fp): fp for fp in file_paths}
                
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"✓ Completed: {Path(file_path).name}")
                    except Exception as e:
                        logger.error(f"✗ Failed: {Path(file_path).name} - {e}")
                        results.append({"file_path": file_path, "error": str(e)})
        else:
            for fp in file_paths:
                try:
                    results.append(self.process_single(fp))
                except Exception as e:
                    logger.error(f"Failed processing {fp}: {e}")
                    results.append({"file_path": fp, "error": str(e)})
        
        return results


# =============================================================================
# CONTENT PARSER - Converts content to structured JSON sections
# =============================================================================

class ContentParser:
    """Parses document content into structured JSON with sections."""
    
    # Section header patterns
    SECTION_PATTERNS = [
        r'(?i)^(?:\d+\.?\s*)?(abstract)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(introduction)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(related\s*work|background|literature)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(method(?:ology)?|approach)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(experiment(?:s)?|results|evaluation)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(discussion)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(conclusion(?:s)?)\s*$',
        r'(?i)^(?:\d+\.?\s*)?(reference(?:s)?|bibliography)\s*$',
    ]
    
    def __init__(self, include_reproducibility_markers: bool = False):
        """
        Initialize parser.
        
        Args:
            include_reproducibility_markers: If True, adds empty reproducibility fields for later analysis
        """
        self.include_reproducibility_markers = include_reproducibility_markers
    
    def parse(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert processed document to structured JSON format."""
        logger.info(f"Parsing content: {processed_data.get('file_name', 'unknown')}")
        
        content = processed_data.get("content", "")
        
        # Extract metadata
        result = {
            "metadata": {
                "file_name": processed_data.get("file_name", ""),
                "file_path": processed_data.get("file_path", ""),
                "extension": processed_data.get("extension", ""),
                "processed_at": processed_data.get("processed_at", ""),
                "parsed_at": datetime.now().isoformat(),
            },
            "content": {
                "title": self._extract_title(content),
                "abstract": self._extract_section(content, "abstract"),
                "sections": self._extract_all_sections(content),
                "full_text": content[:50000] if len(content) > 50000 else content,  # Limit size
            }
        }
        
        # Add reproducibility markers if requested
        if self.include_reproducibility_markers:
            result["reproducibility"] = {
                "has_code": None,
                "has_data": None,
                "github_links": [],
                "datasets_mentioned": [],
                "methodology_clear": None,
                "results_reproducible": None,
                "notes": ""
            }
        
        logger.info(f"Parsed {len(result['content']['sections'])} sections")
        return result
    
    def _extract_title(self, content: str) -> str:
        """Extract title from beginning of content."""
        lines = content.split('\n')[:20]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not re.match(r'(?i)^(abstract|keywords)', line):
                return line
        return ""
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section by name."""
        pattern = rf'(?i)(?:^|\n)(?:\d+\.?\s*)?{section_name}\s*\n+(.*?)(?=\n(?:\d+\.?\s*)?[A-Z][a-z]{{3,}}|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return re.sub(r'\s+', ' ', match.group(1).strip())[:3000]
        return ""
    
    def _extract_all_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract all identifiable sections."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if line is a section header
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                if re.match(pattern, line_stripped):
                    # Save previous section
                    if current_section:
                        sections.append({
                            "title": current_section,
                            "content": re.sub(r'\s+', ' ', '\n'.join(current_content).strip())[:5000]
                        })
                    current_section = line_stripped
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_content.append(line)
        
        # Add last section
        if current_section:
            sections.append({
                "title": current_section,
                "content": re.sub(r'\s+', ' ', '\n'.join(current_content).strip())[:5000]
            })
        
        return sections


# =============================================================================
# JSON SAVER - Saves JSON files to output directory
# =============================================================================

class JSONSaver:
    """Saves parsed documents as JSON files."""
    
    def __init__(self, output_dir: str = "data/jsons"):
        self.output_dir = Path(output_dir)
        self._ensure_dir()
    
    def _ensure_dir(self) -> None:
        """Create output directory if not exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def save(self, parsed_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save parsed data as JSON file."""
        if filename is None:
            original_name = parsed_data.get("metadata", {}).get("file_name", "output")
            filename = Path(original_name).stem + ".json"
        
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Saved: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            raise
    
    def save_batch(self, parsed_data_list: List[Dict[str, Any]]) -> List[str]:
        """Save multiple parsed documents."""
        saved_paths = []
        for data in parsed_data_list:
            try:
                path = self.save(data)
                saved_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to save document: {e}")
        
        logger.info(f"Saved {len(saved_paths)}/{len(parsed_data_list)} files")
        return saved_paths


# =============================================================================
# PIPELINE - Combines all classes for easy use
# =============================================================================

class PDFParsingPipeline:
    """
    Complete pipeline for PDF parsing.
    
    Usage:
        pipeline = PDFParsingPipeline(include_reproducibility=True)
        pipeline.process_file("paper.pdf")
        pipeline.process_directory("data/pdfs")
    """
    
    def __init__(
        self,
        output_dir: str = "data/jsons",
        max_workers: int = 4,
        include_reproducibility: bool = False
    ):
        self.processor = PDFProcessor(max_workers=max_workers)
        self.parser = ContentParser(include_reproducibility_markers=include_reproducibility)
        self.saver = JSONSaver(output_dir=output_dir)
        
        logger.info(f"Pipeline initialized (workers={max_workers}, reproducibility={include_reproducibility})")
    
    def process_file(self, file_path: str) -> str:
        """Process a single file and save as JSON."""
        processed = self.processor.process_single(file_path)
        parsed = self.parser.parse(processed)
        return self.saver.save(parsed)
    
    def process_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        parallel: bool = True
    ) -> List[str]:
        """Process all matching files in a directory."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = list(dir_path.glob(pattern))
        logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
        
        if not files:
            return []
        
        # Process all files
        processed_list = self.processor.process_batch(
            [str(f) for f in files],
            parallel=parallel
        )
        
        # Parse and save each
        saved_paths = []
        for processed in processed_list:
            if "error" not in processed:
                try:
                    parsed = self.parser.parse(processed)
                    path = self.saver.save(parsed)
                    saved_paths.append(path)
                except Exception as e:
                    logger.error(f"Failed to parse/save: {e}")
        
        logger.info(f"Pipeline complete: {len(saved_paths)} files processed")
        return saved_paths


# =============================================================================
# MAIN - CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Parse PDF/documents to JSON")
    arg_parser.add_argument("input", help="File or directory path")
    arg_parser.add_argument("-o", "--output", default="data/jsons", help="Output directory")
    arg_parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers")
    arg_parser.add_argument("-r", "--reproducibility", action="store_true", help="Include reproducibility markers")
    arg_parser.add_argument("-p", "--pattern", default="*.pdf", help="File pattern for directories")
    
    args = arg_parser.parse_args()
    
    pipeline = PDFParsingPipeline(
        output_dir=args.output,
        max_workers=args.workers,
        include_reproducibility=args.reproducibility
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = pipeline.process_file(str(input_path))
        print(f"Output: {result}")
    elif input_path.is_dir():
        results = pipeline.process_directory(str(input_path), pattern=args.pattern)
        print(f"Processed {len(results)} files")
    else:
        print(f"Error: {args.input} not found")