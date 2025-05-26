import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use.agent.service import Agent
from browser_use.browser import BrowserProfile, BrowserSession

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)


async def main():
	browser_session = BrowserSession(
		browser_profile=BrowserProfile(
			trace_path='./tmp/traces/',
			user_data_dir='~/.config/browseruse/profiles/default',
		)
	)

	async with browser_session:
		agent = Agent(
			task='Go to hackernews, then go to apple.com and return all titles of open tabs',
			llm=llm,
			browser_session=browser_session,
		)
		await agent.run()


asyncio.run(main())
