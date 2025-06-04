import asyncio
import pytest 
from pathlib import Path
from browser_use.browser.session import BrowserSession, BrowserProfile
from page_exploration import PageExplorationWorkflow
from langchain_openai import ChatOpenAI
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

@pytest.mark.asyncio 
async def test_page_exploration():
    # Set OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # 替换为你的 API key
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'  # 例如 'http://127.0.0.1:1080'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError('GOOGLE_API_KEY is not set')

    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
    # Setup LLM
    # llm = ChatOpenAI(
    #     temperature=0.7,
    #     model_name="gpt-3.5-turbo"
    # )
    
    # Setup browser session
    profile = BrowserProfile(
        allowed_domains=["water-test.wzsly.cn"],
        headless=False
    )
    browser_session = BrowserSession(browser_profile=profile)
    
    try:
        # Initialize workflow with LLM
        output_dir = Path("exploration_results")
        workflow = PageExplorationWorkflow(
            browser_session=browser_session,
            output_dir=output_dir,
            llm=llm
        )
        
        # Run exploration
        result = await workflow.run(
            start_url="https://water-test.wzsly.cn/",
            max_depth=2
        )
        
        # Verify results
        assert result.pages, "No pages were explored"
        assert result.tree_structure, "No tree structure was created"
        assert result.document, "No documentation was generated"
        
        # Check output files
        assert (output_dir / "documentation.md").exists()
        assert (output_dir / "exploration_data.json").exists()
        
        # Additional assertions for LLM-specific results
        for url, page in result.pages.items():
            assert page.analyzed_elements, f"No element analysis for page {url}"
            assert page.page_purpose, f"No page purpose analysis for {url}"
            
            # Verify element analysis structure
            for element_id, analysis in page.analyzed_elements.items():
                assert 0 <= analysis.importance_score <= 1, "Invalid importance score"
                assert analysis.purpose, "Missing element purpose"
                assert isinstance(analysis.possible_actions, list)
                
            # Verify page purpose structure
            assert page.page_purpose.main_purpose, "Missing main purpose"
            assert isinstance(page.page_purpose.key_features, list)
            assert isinstance(page.page_purpose.user_flows, list)
        
    finally:
        # Cleanup
        await browser_session.stop()

if __name__ == "__main__":
    asyncio.run(test_page_exploration())
