import asyncio
import pytest
from pathlib import Path
from browser_use.browser.session import BrowserSession, BrowserProfile
from page_exploration import PageExplorationWorkflow

@pytest.mark.asyncio
async def test_page_exploration():
    # Setup browser session
    profile = BrowserProfile(
        allowed_domains=["https://www.baidu.com"],  # Add domains you want to test
        headless=False  # Set to True for headless mode
    )
    browser_session = BrowserSession(browser_profile=profile)
    
    try:
        # Initialize workflow
        output_dir = Path("exploration_results")
        workflow = PageExplorationWorkflow(browser_session, output_dir)
        
        # Run exploration
        result = await workflow.run(
            start_url="https://www.baidu.com",
            max_depth=2
        )
        
        # Verify results
        assert result.pages, "No pages were explored"
        assert result.tree_structure, "No tree structure was created"
        assert result.document, "No documentation was generated"
        
        # Check output files
        assert (output_dir / "documentation.md").exists()
        assert (output_dir / "exploration_data.json").exists()
        
    finally:
        # Cleanup
        await browser_session.stop()

if __name__ == "__main__":
    asyncio.run(test_page_exploration()) 