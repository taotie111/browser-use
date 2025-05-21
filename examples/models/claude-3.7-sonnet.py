"""
Simple script that runs the task of opening amazon and searching.
@dev Ensure we have a `ANTHROPIC_API_KEY` variable in our `.env` file.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic

from browser_use import Agent

llm = ChatAnthropic(model_name='claude-3-7-sonnet-20250219', temperature=0.0, timeout=30, stop=None)

agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)


async def main():
	await agent.run(max_steps=10)


asyncio.run(main())
