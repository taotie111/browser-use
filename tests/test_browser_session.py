import asyncio
import base64

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.dom.views import DOMElementNode


class TestBrowserContext:
	"""Tests for browser context functionality using real browser instances."""

	@pytest.fixture(scope='module')
	def event_loop(self):
		"""Create and provide an event loop for async tests."""
		loop = asyncio.get_event_loop_policy().new_event_loop()
		yield loop
		loop.close()

	@pytest.fixture(scope='module')
	def http_server(self):
		"""Create and provide a test HTTP server that serves static content."""
		server = HTTPServer()
		server.start()

		# Add routes for test pages
		server.expect_request('/').respond_with_data(
			'<html><head><title>Test Home Page</title></head><body><h1>Test Home Page</h1><p>Welcome to the test site</p></body></html>',
			content_type='text/html',
		)

		server.expect_request('/scroll_test').respond_with_data(
			"""
            <html>
            <head>
                <title>Scroll Test</title>
                <style>
                    body { height: 3000px; }
                    .marker { position: absolute; }
                    #top { top: 0; }
                    #middle { top: 1000px; }
                    #bottom { top: 2000px; }
                </style>
            </head>
            <body>
                <div id="top" class="marker">Top of the page</div>
                <div id="middle" class="marker">Middle of the page</div>
                <div id="bottom" class="marker">Bottom of the page</div>
            </body>
            </html>
            """,
			content_type='text/html',
		)

		yield server
		server.stop()

	@pytest.fixture
	def base_url(self, http_server):
		"""Return the base URL for the test HTTP server."""
		return f'http://{http_server.host}:{http_server.port}'

	@pytest.fixture(scope='module')
	async def browser_session(self, event_loop):
		"""Create and provide a BrowserSession instance with security disabled."""
		browser_session = BrowserSession(
			# browser_profile=BrowserProfile(...),
			headless=True,
			user_data_dir=None,
		)
		await browser_session.start()
		yield browser_session
		await browser_session.stop()

	def test_is_url_allowed(self):
		"""
		Test the _is_url_allowed method to verify that it correctly checks URLs against
		the allowed domains configuration.
		"""
		# Scenario 1: allowed_domains is None, any URL should be allowed.
		config1 = BrowserProfile(allowed_domains=None)
		context1 = BrowserSession(browser_profile=config1)
		assert context1._is_url_allowed('http://anydomain.com') is True
		assert context1._is_url_allowed('https://anotherdomain.org/path') is True

		# Scenario 2: allowed_domains is provided.
		allowed = ['example.com', '*.mysite.org']
		config2 = BrowserProfile(allowed_domains=allowed)
		context2 = BrowserSession(browser_profile=config2)

		# URL exactly matching
		assert context2._is_url_allowed('http://example.com') is True
		# URL with subdomain (should not be allowed)
		assert context2._is_url_allowed('http://sub.example.com/path') is False
		# URL with different domain (should not be allowed)
		assert context2._is_url_allowed('http://sub.mysite.org') is True
		# URL that matches second allowed domain
		assert context2._is_url_allowed('https://mysite.org/page') is True
		# URL with port number, still allowed (port is stripped)
		assert context2._is_url_allowed('http://example.com:8080') is True
		assert context2._is_url_allowed('https://example.com:443') is True

		# Scenario 3: Malformed URL or empty domain
		# urlparse will return an empty netloc for some malformed URLs.
		assert context2._is_url_allowed('notaurl') is False

	def test_convert_simple_xpath_to_css_selector(self):
		"""
		Test the _convert_simple_xpath_to_css_selector method of BrowserSession.
		This verifies that simple XPath expressions are correctly converted to CSS selectors.
		"""
		# Test empty xpath returns empty string
		assert BrowserSession._convert_simple_xpath_to_css_selector('') == ''

		# Test a simple xpath without indices
		xpath = '/html/body/div/span'
		expected = 'html > body > div > span'
		result = BrowserSession._convert_simple_xpath_to_css_selector(xpath)
		assert result == expected

		# Test xpath with an index on one element: [2] should translate to :nth-of-type(2)
		xpath = '/html/body/div[2]/span'
		expected = 'html > body > div:nth-of-type(2) > span'
		result = BrowserSession._convert_simple_xpath_to_css_selector(xpath)
		assert result == expected

		# Test xpath with indices on multiple elements
		xpath = '/ul/li[3]/a[1]'
		expected = 'ul > li:nth-of-type(3) > a:nth-of-type(1)'
		result = BrowserSession._convert_simple_xpath_to_css_selector(xpath)
		assert result == expected

	def test_enhanced_css_selector_for_element(self):
		"""
		Test the _enhanced_css_selector_for_element method to verify that
		it returns the correct CSS selector string for a DOMElementNode.
		"""
		# Create a DOMElementNode instance with a complex set of attributes
		dummy_element = DOMElementNode(
			tag_name='div',
			is_visible=True,
			parent=None,
			xpath='/html/body/div[2]',
			attributes={'class': 'foo bar', 'id': 'my-id', 'placeholder': 'some "quoted" text', 'data-testid': '123'},
			children=[],
		)

		# Call the method with include_dynamic_attributes=True
		actual_selector = BrowserSession._enhanced_css_selector_for_element(dummy_element, include_dynamic_attributes=True)

		# Expected conversion includes the xpath conversion, class attributes, and other attributes
		expected_selector = (
			'html > body > div:nth-of-type(2).foo.bar[id="my-id"][placeholder*="some \\"quoted\\" text"][data-testid="123"]'
		)
		assert actual_selector == expected_selector, f'Expected {expected_selector}, but got {actual_selector}'

	@pytest.mark.asyncio
	async def test_navigate_and_get_current_page(self, browser_session, base_url):
		"""Test that navigate method changes the URL and get_current_page returns the proper page."""
		# Navigate to the test page
		await browser_session.navigate(f'{base_url}/')

		# Get the current page
		page = await browser_session.get_current_page()

		# Verify the page URL matches what we navigated to
		assert f'{base_url}/' in page.url

		# Verify the page title
		title = await page.title()
		assert title == 'Test Home Page'

	@pytest.mark.asyncio
	async def test_refresh_page(self, browser_session, base_url):
		"""Test that refresh_page correctly reloads the current page."""
		# Navigate to the test page
		await browser_session.navigate(f'{base_url}/')

		# Get the current page before refresh
		page_before = await browser_session.get_current_page()

		# Refresh the page
		await browser_session.refresh()

		# Get the current page after refresh
		page_after = await browser_session.get_current_page()

		# Verify it's still on the same URL
		assert page_after.url == page_before.url

		# Verify the page title is still correct
		title = await page_after.title()
		assert title == 'Test Home Page'

	@pytest.mark.asyncio
	async def test_execute_javascript(self, browser_session, base_url):
		"""Test that execute_javascript correctly executes JavaScript in the current page."""
		# Navigate to a test page
		await browser_session.navigate(f'{base_url}/')

		# Execute a simple JavaScript snippet that returns a value
		result = await browser_session.execute_javascript('document.title')

		# Verify the result
		assert result == 'Test Home Page'

		# Execute JavaScript that modifies the page
		await browser_session.execute_javascript("document.body.style.backgroundColor = 'red'")

		# Verify the change by reading back the value
		bg_color = await browser_session.execute_javascript('document.body.style.backgroundColor')
		assert bg_color == 'red'

	@pytest.mark.asyncio
	async def test_get_scroll_info(self, browser_session, base_url):
		"""Test that get_scroll_info returns the correct scroll position information."""
		# Navigate to the scroll test page
		await browser_session.navigate(f'{base_url}/scroll_test')
		page = await browser_session.get_current_page()

		# Get initial scroll info
		pixels_above_initial, pixels_below_initial = await browser_session.get_scroll_info(page)

		# Verify initial scroll position
		assert pixels_above_initial == 0, 'Initial scroll position should be at the top'
		assert pixels_below_initial > 0, 'There should be content below the viewport'

		# Scroll down the page
		await browser_session.execute_javascript('window.scrollBy(0, 500)')
		await asyncio.sleep(0.2)  # Brief delay for scroll to complete

		# Get new scroll info
		pixels_above_after_scroll, pixels_below_after_scroll = await browser_session.get_scroll_info(page)

		# Verify new scroll position
		assert pixels_above_after_scroll > 0, 'Page should be scrolled down'
		assert pixels_above_after_scroll >= 400, 'Page should be scrolled down at least 400px'
		assert pixels_below_after_scroll < pixels_below_initial, 'Less content should be below viewport after scrolling'

	@pytest.mark.asyncio
	async def test_take_screenshot(self, browser_session, base_url):
		"""Test that take_screenshot returns a valid base64 encoded image."""
		# Navigate to the test page
		await browser_session.navigate(f'{base_url}/')

		# Take a screenshot
		screenshot_base64 = await browser_session.take_screenshot()

		# Verify the screenshot is a valid base64 string
		assert isinstance(screenshot_base64, str)
		assert len(screenshot_base64) > 0

		# Verify it can be decoded as base64
		try:
			image_data = base64.b64decode(screenshot_base64)
			# Verify the data starts with a valid image signature (PNG file header)
			assert image_data[:8] == b'\x89PNG\r\n\x1a\n', 'Screenshot is not a valid PNG image'
		except Exception as e:
			pytest.fail(f'Failed to decode screenshot as base64: {e}')

	@pytest.mark.asyncio
	async def test_switch_tab_operations(self, browser_session, base_url):
		"""Test tab creation, switching, and closing operations."""
		# Navigate to home page in first tab
		await browser_session.navigate(f'{base_url}/')

		# Create a new tab
		await browser_session.create_new_tab(f'{base_url}/scroll_test')

		# Verify we have two tabs now
		tabs_info = await browser_session.get_tabs_info()
		assert len(tabs_info) == 2, 'Should have two tabs open'

		# Verify current tab is the scroll test page
		current_page = await browser_session.get_current_page()
		assert f'{base_url}/scroll_test' in current_page.url

		# Switch back to the first tab
		await browser_session.switch_to_tab(0)

		# Verify we're back on the home page
		current_page = await browser_session.get_current_page()
		assert f'{base_url}/' in current_page.url

		# Close the second tab
		await browser_session.close_tab(1)

		# Verify we only have one tab left
		tabs_info = await browser_session.get_tabs_info()
		assert len(tabs_info) == 1, 'Should have one tab open after closing the second'

	@pytest.mark.asyncio
	async def test_remove_highlights(self, browser_session, base_url):
		"""Test that remove_highlights successfully removes highlight elements."""
		# Navigate to a test page
		await browser_session.navigate(f'{base_url}/')

		# Add a highlight via JavaScript
		await browser_session.execute_javascript("""
            const container = document.createElement('div');
            container.id = 'playwright-highlight-container';
            document.body.appendChild(container);
            
            const highlight = document.createElement('div');
            highlight.id = 'playwright-highlight-1';
            container.appendChild(highlight);
            
            const element = document.querySelector('h1');
            element.setAttribute('browser-user-highlight-id', 'playwright-highlight-1');
        """)

		# Verify the highlight container exists
		container_exists = await browser_session.execute_javascript(
			"document.getElementById('playwright-highlight-container') !== null"
		)
		assert container_exists, 'Highlight container should exist before removal'

		# Call remove_highlights
		await browser_session.remove_highlights()

		# Verify the highlight container was removed
		container_exists_after = await browser_session.execute_javascript(
			"document.getElementById('playwright-highlight-container') !== null"
		)
		assert not container_exists_after, 'Highlight container should be removed'

		# Verify the highlight attribute was removed from the element
		attribute_exists = await browser_session.execute_javascript(
			"document.querySelector('h1').hasAttribute('browser-user-highlight-id')"
		)
		assert not attribute_exists, 'browser-user-highlight-id attribute should be removed'
