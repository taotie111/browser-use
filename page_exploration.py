from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import asyncio
from datetime import datetime

from browser_use.dom.views import DOMElementNode, SelectorMap
from browser_use.dom.history_tree_processor.view import DOMHistoryElement
from browser_use.dom.clickable_element_processor.service import ClickableElementProcessor
from browser_use.browser.session import BrowserSession
from browser_use.agent.views import ActionResult

@dataclass
class PageNode:
    """Represents a page in the exploration tree"""
    url: str
    title: str
    parent_url: Optional[str]
    clickable_elements: list['DOMHistoryElement']  # Forward reference for type hinting
    notes: dict[str, str]  # Element index -> note
    timestamp: str

@dataclass
class ExplorationResult:
    """Results of the page exploration"""
    pages: dict[str, PageNode]  # URL -> PageNode
    tree_structure: dict[str, list[str]]  # URL -> list of child URLs
    document: str  # Generated documentation

class PageExplorationWorkflow:
    """
    Implements a workflow for exploring web pages by:
    1. Recording clickable elements
    2. Creating notes about functionality
    3. Building a tree structure of pages
    4. Generating documentation
    """
    
    def __init__(self, browser_session: BrowserSession, output_dir: Path):
        self.browser_session = browser_session
        self.output_dir = output_dir
        self.pages: dict[str, PageNode] = {}
        self.tree_structure: dict[str, list[str]] = {}
        
    async def explore_page(self, url: str, parent_url: Optional[str] = None) -> None:
        """
        Explores a single page, recording its elements and structure
        """
        # Skip if already explored
        if url in self.pages:
            return
            
        # Navigate to page
        page = await self.browser_session.get_current_page()
        if page.url != url:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            
        # Get page state
        state_summary = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)
        if not state_summary:
            return
            
        # Record clickable elements
        clickable_elements = []
        for idx, element in state_summary.selector_map.items():
            if isinstance(element, DOMElementNode):
                history_element = self._convert_to_history_element(element)
                clickable_elements.append(history_element)
                
        # Create page node
        page_node = PageNode(
            url=url,
            title=await page.title(),
            parent_url=parent_url,
            clickable_elements=clickable_elements,
            notes={},
            timestamp=datetime.now().isoformat()
        )
        
        # Update structures
        self.pages[url] = page_node
        if parent_url:
            if parent_url not in self.tree_structure:
                self.tree_structure[parent_url] = []
            self.tree_structure[parent_url].append(url)
            
    def _convert_to_history_element(self, element: DOMElementNode) -> DOMHistoryElement:
        """Convert DOMElementNode to DOMHistoryElement for storage"""
        from browser_use.dom.history_tree_processor.service import HistoryTreeProcessor
        return HistoryTreeProcessor.convert_dom_element_to_history_element(element)
        
    async def explore_recursively(self, start_url: str, max_depth: int = 3) -> None:
        """
        Recursively explores pages starting from a URL up to max_depth
        """
        async def explore(url: str, depth: int, parent_url: Optional[str] = None):
            if depth > max_depth:
                return
                
            await self.explore_page(url, parent_url)
            
            # Get clickable elements that lead to new pages
            page_node = self.pages[url]
            for element in page_node.clickable_elements:
                if element.attributes.get("href"):
                    next_url = element.attributes["href"]
                    if next_url.startswith(start_url):  # Stay within same domain
                        await explore(next_url, depth + 1, url)
                        
        await explore(start_url, 1)
        
    def add_element_note(self, url: str, element_index: int, note: str) -> None:
        """Add a note about a specific element's functionality"""
        if url in self.pages:
            self.pages[url].notes[str(element_index)] = note
            
    def generate_documentation(self) -> str:
        """
        Generates a markdown document describing the explored pages
        """
        doc = ["# Website Structure Documentation\n"]
        
        def add_page_to_doc(url: str, depth: int = 0):
            page = self.pages[url]
            indent = "  " * depth
            
            # Add page header
            doc.append(f"{indent}## {page.title}\n")
            doc.append(f"{indent}URL: {page.url}\n")
            
            # Add clickable elements
            doc.append(f"{indent}### Interactive Elements:\n")
            for element in page.clickable_elements:
                element_id = element.highlight_index
                note = page.notes.get(str(element_id), "")
                
                # Format element info
                element_info = f"{indent}- {element.tag_name}"
                if element.attributes.get("id"):
                    element_info += f" (id: {element.attributes['id']})"
                if element.attributes.get("class"):
                    element_info += f" (class: {element.attributes['class']})"
                if note:
                    element_info += f"\n{indent}  Note: {note}"
                doc.append(element_info + "\n")
                
            # Add child pages
            if url in self.tree_structure:
                doc.append(f"{indent}### Child Pages:\n")
                for child_url in self.tree_structure[url]:
                    add_page_to_doc(child_url, depth + 1)
                    
        # Start with root pages (those without parents)
        root_pages = [url for url, page in self.pages.items() if not page.parent_url]
        for root_url in root_pages:
            add_page_to_doc(root_url)
            
        return "\n".join(doc)
        
    async def save_results(self) -> None:
        """
        Saves exploration results to files
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documentation
        doc = self.generate_documentation()
        doc_path = self.output_dir / "documentation.md"
        doc_path.write_text(doc)
        
        # Save raw data
        data = {
            "pages": {url: self._convert_page_to_serializable(page) for url, page in self.pages.items()},  # New Start
            "tree_structure": self.tree_structure
        }
        print("Data structure before JSON serialization:")
        print(self.tree_structure)  # Print the formatted JSON string
        print(self.pages)  # P
        data_path = self.output_dir / "exploration_data.json"
        data_path.write_text(json.dumps(data, indent=2))
        
    def _convert_page_to_serializable(self, page: PageNode) -> dict:  # New Start
        """Convert PageNode to a serializable format."""
        return {
            "url": page.url,
            "title": page.title,
            "parent_url": page.parent_url,
            "clickable_elements": [self._convert_history_element_to_dict(elem) for elem in page.clickable_elements],  # New Start
            "notes": page.notes,
            "timestamp": page.timestamp
        }  # New End

    def _convert_history_element_to_dict(self, element: DOMHistoryElement) -> dict:  # New Start
        """Convert DOMHistoryElement to a serializable format."""
        return {
            "tag_name": element.tag_name,
            "attributes": element.attributes,
            "highlight_index": element.highlight_index,
            # Add other necessary fields for serialization here
        }  # New End
        
    async def run(self, start_url: str, max_depth: int = 3) -> ExplorationResult:
        """
        Runs the complete page exploration workflow
        """
        # Explore pages
        await self.explore_recursively(start_url, max_depth)
        
        # Generate and save results
        await self.save_results()
        
        return ExplorationResult(
            pages=self.pages,
            tree_structure=self.tree_structure,
            document=self.generate_documentation()
        ) 
