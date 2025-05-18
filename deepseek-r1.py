import asyncio
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from browser_use import Agent

# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')
# if not api_key:
# 	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=('go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result'),
		llm=ChatDeepSeek(
			base_url='http://172.16.1.9:11434/v1',
			model='deepseek-r1:70b',
			api_key=SecretStr(api_key),
		),
		use_vision=True,
		max_failures=2,
		max_actions_per_step=1,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())
