import asyncio
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession
# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')
# if not api_key:
# 	raise ValueError('DEEPSEEK_API_KEY is not set')

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		headless=False,
		viewport_expansion=0,
		user_data_dir='~/.config/browseruse/profiles/default',
	)
)
async def run_search():
	agent = Agent(
		task="""
		0. 验证码登录（通用验证码 114477）
			市局帐号：15794791975
			法人帐号：15718828461  
			专家帐号：18958811221
		1. 打开网址https://slgcjg.wzsly.cn/login?redirect=/screen/homePage

		""",
		llm=ChatDeepSeek(
			base_url='http://172.16.1.9:11434/v1',
			model='deepseek-chat',
			api_key=SecretStr(api_key),
		),
		use_vision=True,
		max_failures=2,
		max_actions_per_step=1,
		browser_session=browser_session,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())
