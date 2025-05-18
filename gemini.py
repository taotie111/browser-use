import asyncio
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
	raise ValueError('GEMINI_API_KEY is not set')
print(f'GEMINI_API_KEY: {api_key}')
llm =  ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key="AIzaSyDw8wDt0oP7oR_xFaecMyJ-Ub22ETreI7g"  # 替换为你的API密钥
    )
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'  # 例如 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

browser = Browser(
	config=BrowserConfig(
		new_context_config=BrowserContextConfig(
			viewport_expansion=0,
		)
	)
)


async def run_search():
	agent = Agent(
		task="""1. 打开网址https://www.bilibili.com/
		2. 搜索莲花池相关内容
		3. 点开最新视频
		4. 查看下方评论
		5. 总结评论内容
		6. 生成一段关于莲花池的介绍""",
		llm=llm,
		max_actions_per_step=4,
		browser=browser,
	)

	await agent.run(max_steps=25)


if __name__ == '__main__':
	asyncio.run(run_search())
