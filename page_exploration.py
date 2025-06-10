from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import json
import asyncio
from datetime import datetime
import base64
from PIL import Image
import io
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from playwright.async_api import TimeoutError
# 新的推荐导入方式
from langchain_openai import ChatOpenAI
from google.api_core.exceptions import FailedPrecondition
from langchain_deepseek import ChatDeepSeek
import os

from browser_use.dom.views import DOMElementNode, SelectorMap
from browser_use.dom.history_tree_processor.view import DOMHistoryElement
from browser_use.dom.clickable_element_processor.service import ClickableElementProcessor
from browser_use.browser.session import BrowserSession
from browser_use.agent.views import ActionResult

@dataclass
class ElementAnalysis:
    """Represents LLM's analysis of a page element"""
    element_id: str
    element_type: str
    purpose: str
    possible_actions: List[str]
    importance_score: float  # 0-1 score indicating element importance
    interaction_hints: List[str]
    related_elements: List[str]  # IDs of related elements

@dataclass
class PagePurpose:
    """Represents the analyzed purpose of a page"""
    main_purpose: str
    key_features: List[str]
    ui_elements_summary: str
    user_flows: List[str]  # Common user interaction flows
    key_interaction_points: List[str]  # Important interaction areas

@dataclass
class PageNode:
    """Represents a page in the exploration tree"""
    url: str
    title: str
    parent_url: Optional[str]
    clickable_elements: list['DOMHistoryElement']
    selected_elements: list['DOMHistoryElement']
    analyzed_elements: dict[str, ElementAnalysis]  # Element ID -> Analysis
    notes: dict[str, str]
    timestamp: str
    screenshot: Optional[str]
    page_purpose: Optional[PagePurpose]

@dataclass
class ExplorationResult:
    """Results of the page exploration"""
    pages: dict[str, PageNode]  # URL -> PageNode
    tree_structure: dict[str, list[str]]  # URL -> list of child URLs
    document: str  # Generated documentation

class PageExplorationWorkflow:
    """
    Implements a workflow for exploring web pages using LLM analysis
    """
    
    def __init__(self, 
                 browser_session: BrowserSession, 
                 output_dir: Path,
                 llm: BaseChatModel):
        self.browser_session = browser_session
        self.output_dir = output_dir
        self.pages: dict[str, PageNode] = {}
        self.tree_structure: dict[str, list[str]] = {}
        self.llm = llm
        self.llm_switched_to_deepseek = False
        
    async def _invoke_llm(self, messages: List[HumanMessage]):
        """
        purpose: Invokes the language model, with a fallback from Gemini to DeepSeek if a location-based error occurs.
        params:
            messages (List[HumanMessage]): The list of messages to send to the LLM.
        returns:
            The response from the language model.
        """
        try:
            return await self.llm.ainvoke(messages)
        except FailedPrecondition as e:
            if "User location is not supported" in str(e) and not self.llm_switched_to_deepseek:
                print("Gemini API location not supported, attempting to switch to DeepSeek...")
                api_key = os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    print("DEEPSEEK_API_KEY environment variable not found. Cannot switch LLM.")
                    raise e
                
                self.llm = ChatDeepSeek(
                    model="deepseek-chat", 
                    api_key=api_key
                )
                self.llm_switched_to_deepseek = True
                print("Successfully switched to DeepSeek model.")
                return await self.llm.ainvoke(messages)
            else:
                raise e

    def _extract_json_from_llm_response(self, content: str) -> str:
        """
        purpose: Extracts a JSON string from the LLM's response, which may be wrapped in markdown code blocks.
        params:
            content (str): The string content from the LLM response.
        returns:
            str: The extracted JSON string.
        """
        # Handle case where the response is wrapped in ```json ... ```
        if "```json" in content:
            return content.split("```json")[1].split("```")[0].strip()
        # Handle case where the response is wrapped in ``` ... ```
        elif "```" in content:
            # This will take the content from the first ``` to the second ```
            return content.split("```")[1].strip()
        # Otherwise, assume the response is the JSON content itself
        return content

    def _is_important_element(self, element: DOMElementNode) -> bool:
        """Determine if a non-clickable element is important enough to include"""
        important_tags = {'h1', 'h2', 'h3', 'form', 'nav', 'main', 'header', 'footer'}
        important_roles = {'banner', 'main', 'navigation', 'search', 'form'}
        
        # Check tag name
        if element.tag_name.lower() in important_tags:
            return True
            
        # Check ARIA role
        if element.attributes.get('role', '').lower() in important_roles:
            return True
            
        # Check for important form elements
        if element.tag_name.lower() in {'input', 'select', 'textarea'}:
            return True
            
        # Check for elements with important attributes
        important_attrs = {'aria-label', 'data-testid', 'data-cy', 'data-qa'}
        if any(attr in element.attributes for attr in important_attrs):
            return True
            
        return False

    async def _analyze_element_with_llm(self, 
                                      element: DOMElementNode, 
                                      page_context: str,
                                      surrounding_elements: List[DOMElementNode]) -> Optional[ElementAnalysis]:
        """Use LLM to analyze a single element in context"""
        
        # Prepare element context
        element_info = {
            "tag": element.tag_name,
            "attributes": element.attributes,
            "text_content": getattr(element, 'text_content', ''),
            "is_clickable": element.is_clickable if hasattr(element, 'is_clickable') else False
        }
        
        # Prepare surrounding elements context
        surrounding_context = []
        for elem in surrounding_elements:
            surrounding_context.append({
                "tag": elem.tag_name,
                "text": getattr(elem, 'text_content', ''),
                "relation": "nearby"  # Could be enhanced with spatial/DOM relationship
            })

        # Create prompt for LLM
        prompt = f"""Analyze this web page element in context:

                    Element: {json.dumps(element_info, indent=2)}

                    Page Context: {page_context}

                    Surrounding Elements: {json.dumps(surrounding_context, indent=2)}

                    Analyze the element and provide:
                    1. Element's purpose
                    2. Possible user interactions
                    3. Importance (0-1 score)
                    4. Related elements
                    5. Interaction hints

                    Format your response as JSON with these keys:
                    - purpose: string
                    - possible_actions: list of strings
                    - importance_score: float
                    - related_elements: list of strings
                    - interaction_hints: list of strings
                """
        # Get LLM analysis
        response = await self._invoke_llm([HumanMessage(content=prompt)])
        print(response, "response")
        json_content = self._extract_json_from_llm_response(response.content)
        try:
            analysis = json.loads(json_content)
        except json.JSONDecodeError:
            error_dir = self.output_dir / "llm_json_errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().isoformat().replace(":", "-").replace(".", "-")
            error_file = error_dir / f"element_analysis_error_{ts}.json"
            error_file.write_text(json_content)
            print(f"Failed to decode JSON for element analysis. Content saved to {error_file}")
            print(f"Problematic JSON content was: {json_content}")
            return None
        
        return ElementAnalysis(
            element_id=str(element.highlight_index),
            element_type=element.tag_name,
            purpose=analysis['purpose'],
            possible_actions=analysis['possible_actions'],
            importance_score=analysis['importance_score'],
            interaction_hints=analysis['interaction_hints'],
            related_elements=analysis['related_elements']
        )

    async def _analyze_page_purpose(self, 
                                  page_elements: list[DOMHistoryElement], 
                                  screenshot: str,
                                  analyzed_elements: dict[str, ElementAnalysis]) -> Optional[PagePurpose]:
        """Enhanced page purpose analysis using LLM"""
        
        # Prepare page context
        page_context = {
            "elements": [
                {
                    "type": elem.tag_name,
                    "text": getattr(elem, 'text_content', ''),
                    "analysis": analyzed_elements.get(str(elem.highlight_index))
                }
                for elem in page_elements
            ],
            "has_screenshot": bool(screenshot)
        }

        prompt = f"""Analyze this web page structure and provide:
            1. Main purpose of the page
            2. Key features available
            3. Summary of UI elements
            4. Common user interaction flows
            5. Key interaction points

            Page Context: {json.dumps(page_context, indent=2)}

            Format your response as JSON with these keys:
            - main_purpose: string
            - key_features: list of strings
            - ui_elements_summary: string
            - user_flows: list of strings
            - key_interaction_points: list of strings
            """

        response = await self._invoke_llm([HumanMessage(content=prompt)])
        json_content = self._extract_json_from_llm_response(response.content)
        try:
            analysis = json.loads(json_content)
        except json.JSONDecodeError:
            error_dir = self.output_dir / "llm_json_errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().isoformat().replace(":", "-").replace(".", "-")
            error_file = error_dir / f"page_purpose_error_{ts}.json"
            error_file.write_text(json_content)
            print(f"Failed to decode JSON for page purpose analysis. Content saved to {error_file}")
            print(f"Problematic JSON content was: {json_content}")
            return None
        
        return PagePurpose(
            main_purpose=analysis['main_purpose'],
            key_features=analysis['key_features'],
            ui_elements_summary=analysis['ui_elements_summary'],
            user_flows=analysis['user_flows'],
            key_interaction_points=analysis['key_interaction_points']
        )

    async def explore_page(self, url: str, parent_url: Optional[str] = None) -> None:
        """
        Explores a single page using LLM analysis
        """
        if url in self.pages:
            return
            
        page = await self.browser_session.get_current_page()
        if page.url != url:
            await page.goto(url)
            try:
                await page.wait_for_load_state("networkidle")
            except TimeoutError:
                print(f"Timeout waiting for page {url} to be idle. Continuing with current state.")
            
        state_summary = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=True)
        if not state_summary:
            return
            
        screenshot = await self.browser_session.take_screenshot(full_page=True)
        
        # Analyze all elements with LLM
        analyzed_elements = {}
        all_elements = list(state_summary.selector_map.values())
        
        for idx, element in state_summary.selector_map.items():
            if isinstance(element, DOMElementNode):
                # Get nearby elements for context
                surrounding_elements = self._get_surrounding_elements(element, all_elements)
                
                # Get LLM analysis
                analysis = await self._analyze_element_with_llm(
                    element,
                    f"Page title: {await page.title()}\nURL: {url}",
                    surrounding_elements
                )
                if analysis:
                    analyzed_elements[str(idx)] = analysis

        # Separate elements based on analysis
        clickable_elements = []
        selected_elements = []
        
        for idx, element in state_summary.selector_map.items():
            if isinstance(element, DOMElementNode):
                history_element = self._convert_to_history_element(element)
                analysis = analyzed_elements.get(str(idx))
                is_clickable = getattr(element, 'is_clickable', False)

                if is_clickable or (analysis and analysis.importance_score > 0.7):
                    clickable_elements.append(history_element)
                elif analysis and analysis.importance_score > 0.4:
                    selected_elements.append(history_element)

        # Get comprehensive page analysis
        page_purpose = await self._analyze_page_purpose(
            clickable_elements + selected_elements,
            screenshot,
            analyzed_elements
        )
                
        page_node = PageNode(
            url=url,
            title=await page.title(),
            parent_url=parent_url,
            clickable_elements=clickable_elements,
            selected_elements=selected_elements,
            analyzed_elements=analyzed_elements,
            notes={},
            timestamp=datetime.now().isoformat(),
            screenshot=screenshot,
            page_purpose=page_purpose
        )
        
        self.pages[url] = page_node
        if parent_url:
            if parent_url not in self.tree_structure:
                self.tree_structure[parent_url] = []
            self.tree_structure[parent_url].append(url)

    def _get_surrounding_elements(self, 
                                target: DOMElementNode, 
                                all_elements: List[DOMElementNode],
                                context_size: int = 5) -> List[DOMElementNode]:
        """Get nearby elements for context"""
        if not hasattr(target, 'highlight_index'):
            return []
            
        target_idx = target.highlight_index
        return [
            elem for elem in all_elements 
            if hasattr(elem, 'highlight_index') 
            and abs(elem.highlight_index - target_idx) <= context_size
        ]

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
        """Enhanced documentation generation including LLM analysis"""
        doc = ["# Website Structure Documentation\n"]
        
        def add_page_to_doc(url: str, depth: int = 0):
            page = self.pages[url]
            indent = "  " * depth
            
            # Add page header
            doc.append(f"{indent}## {page.title}\n")
            doc.append(f"{indent}URL: {page.url}\n")
            
            # Add page purpose analysis
            if page.page_purpose:
                doc.append(f"{indent}### Page Analysis\n")
                doc.append(f"{indent}**Main Purpose**: {page.page_purpose.main_purpose}\n")
                doc.append(f"{indent}**Key Features**:\n")
                for feature in page.page_purpose.key_features:
                    doc.append(f"{indent}- {feature}\n")
                doc.append(f"{indent}**Common User Flows**:\n")
                for flow in page.page_purpose.user_flows:
                    doc.append(f"{indent}- {flow}\n")
                doc.append(f"{indent}**Key Interaction Points**:\n")
                for point in page.page_purpose.key_interaction_points:
                    doc.append(f"{indent}- {point}\n")
                doc.append(f"{indent}**UI Summary**:\n{indent}{page.page_purpose.ui_elements_summary}\n")
            
            # Add analyzed elements
            doc.append(f"{indent}### Element Analysis:\n")
            for element_id, analysis in page.analyzed_elements.items():
                doc.append(f"{indent}#### Element {element_id} ({analysis.element_type})\n")
                doc.append(f"{indent}- Purpose: {analysis.purpose}\n")
                doc.append(f"{indent}- Possible Actions:\n")
                for action in analysis.possible_actions:
                    doc.append(f"{indent}  - {action}\n")
                doc.append(f"{indent}- Importance Score: {analysis.importance_score:.2f}\n")
                if analysis.interaction_hints:
                    doc.append(f"{indent}- Interaction Hints:\n")
                    for hint in analysis.interaction_hints:
                        doc.append(f"{indent}  - {hint}\n")
                
            # Add child pages
            if url in self.tree_structure:
                doc.append(f"{indent}### Child Pages:\n")
                for child_url in self.tree_structure[url]:
                    add_page_to_doc(child_url, depth + 1)
                    
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
            "pages": {url: self._convert_page_to_serializable(page) for url, page in self.pages.items()},
            "tree_structure": self.tree_structure
        }
        print("Data structure before JSON serialization:")
        print(self.tree_structure)  # Print the formatted JSON string
        print(self.pages)  # P
        data_path = self.output_dir / "exploration_data.json"
        data_path.write_text(json.dumps(data, indent=2))
        
    def _convert_page_to_serializable(self, page: PageNode) -> dict:
        """Convert PageNode to a serializable format with LLM analysis"""
        return {
            "url": page.url,
            "title": page.title,
            "parent_url": page.parent_url,
            "clickable_elements": [self._convert_history_element_to_dict(elem) for elem in page.clickable_elements],
            "selected_elements": [self._convert_history_element_to_dict(elem) for elem in page.selected_elements],
            "analyzed_elements": {
                element_id: {
                    "element_type": analysis.element_type,
                    "purpose": analysis.purpose,
                    "possible_actions": analysis.possible_actions,
                    "importance_score": analysis.importance_score,
                    "interaction_hints": analysis.interaction_hints,
                    "related_elements": analysis.related_elements
                }
                for element_id, analysis in page.analyzed_elements.items()
            },
            "notes": page.notes,
            "timestamp": page.timestamp,
            "screenshot": page.screenshot,
            "page_purpose": {
                "main_purpose": page.page_purpose.main_purpose,
                "key_features": page.page_purpose.key_features,
                "ui_elements_summary": page.page_purpose.ui_elements_summary,
                "user_flows": page.page_purpose.user_flows,
                "key_interaction_points": page.page_purpose.key_interaction_points
            } if page.page_purpose else None
        }

    def _convert_history_element_to_dict(self, element: DOMHistoryElement) -> dict:
        """Convert DOMHistoryElement to a serializable format."""
        return {
            "tag_name": element.tag_name,
            "attributes": element.attributes,
            "highlight_index": element.highlight_index,
            "text_content": element.text_content if hasattr(element, 'text_content') else None
        }
        
    async def run(self, start_url: str, max_depth: int = 3) -> ExplorationResult:
        """
        Runs the complete page exploration workflow with LLM analysis
        """
        await self.explore_recursively(start_url, max_depth)
        await self.save_results()
        
        return ExplorationResult(
            pages=self.pages,
            tree_structure=self.tree_structure,
            document=self.generate_documentation()
        ) 
