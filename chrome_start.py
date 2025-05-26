import asyncio
import logging
import os
from pathlib import Path
import psutil

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('browser_startup')


class BrowserProfile:
    """
    简化版的浏览器配置类，模拟你代码中的BrowserProfile部分配置
    """

    def __init__(self):
        # 这里模拟一些配置项
        self.user_data_dir = Path('./my_user_data_dir')
        self.headless = False
        self.viewport = {'width': 1280, 'height': 720}
        self.channel = 'chrome'  # 你可以指定 chrome、chromium、msedge 等
        self.launch_args = {
            'headless': self.headless,
            'channel': self.channel,
            'args': [
                # 一些常用启动参数，比如禁用沙箱等
                '--disable-gpu',
                '--no-sandbox',
            ],
            # 视口大小可以在new_context时设置
        }

    def prepare_user_data_dir(self):
        # 创建用户数据目录
        if not self.user_data_dir.exists():
            logger.info(f'Creating user data dir at {self.user_data_dir}')
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f'User data dir already exists: {self.user_data_dir}')


async def launch_browser(browser_profile: BrowserProfile):
    """
    启动浏览器的核心方法，参考了你代码中的 setup_playwright 和 setup_browser_context
    """
    # 1. 启动 playwright
    playwright = await async_playwright().start()
    logger.info('Playwright started')

    # 2. 判断是否使用 user_data_dir 来启动持久化上下文
    browser_context = None
    browser = None

    if browser_profile.user_data_dir.exists():
        browser_profile.prepare_user_data_dir()
        logger.info(f'Launching persistent Chromium with user_data_dir: {browser_profile.user_data_dir}')
        browser_context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(browser_profile.user_data_dir),
            headless=browser_profile.headless,
            channel=browser_profile.channel,
            args=browser_profile.launch_args.get('args', []),
            viewport=browser_profile.viewport,
        )
        # Playwright 目前launch_persistent_context返回的browser对象可能为None
        browser = browser_context.browser
    else:
        logger.info('Launching temporary Chromium (no user data dir)')
        browser = await playwright.chromium.launch(
            headless=browser_profile.headless,
            channel=browser_profile.channel,
            args=browser_profile.launch_args.get('args', []),
        )
        browser_context = await browser.new_context(viewport=browser_profile.viewport)

    # 3. 检测新启动的chrome进程PID（模拟你的代码）
    current_process = psutil.Process(os.getpid())
    child_pids_before = {p.pid for p in current_process.children(recursive=True)}
    # 因为启动已经完成了，这里主要是示意，实际要放到启动前后比较

    # 4. 打开一个新页面测试
    page = await browser_context.new_page() if not browser_context.pages else browser_context.pages[0]
    await page.goto('https://example.com')
    logger.info(f'Page opened: {page.url}')

    # 5. 等待几秒，保持浏览器打开
    await asyncio.sleep(10)

    # 6. 关闭
    await browser_context.close()
    if browser and browser.is_connected():
        await browser.close()
    await playwright.stop()
    logger.info('Browser and Playwright stopped')


async def main():
    profile = BrowserProfile()
    profile.prepare_user_data_dir()
    await launch_browser(profile)


if __name__ == '__main__':
    asyncio.run(main())
