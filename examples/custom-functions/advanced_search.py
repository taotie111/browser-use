import asyncio
import http
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.profile import BrowserProfile

logger = logging.getLogger(__name__)


class Person(BaseModel):
	name: str
	email: str | None = None


class PersonList(BaseModel):
	people: list[Person]


SERP_API_KEY = os.getenv('SERPER_API_KEY')
if not SERP_API_KEY:
	raise ValueError('SERPER_API_KEY is not set')

controller = Controller(exclude_actions=['search_google'], output_model=PersonList)


@controller.registry.action('Search the web for a specific query')
async def search_web(query: str):
	# do a serp search for the query
	conn = http.client.HTTPSConnection('google.serper.dev')
	payload = json.dumps({'q': query})
	headers = {'X-API-KEY': SERP_API_KEY, 'Content-Type': 'application/json'}
	conn.request('POST', '/search', payload, headers)
	res = conn.getresponse()
	data = res.read()
	serp_data = json.loads(data.decode('utf-8'))

	# exclude searchParameters and credits
	serp_data = {k: v for k, v in serp_data.items() if k not in ['searchParameters', 'credits']}

	# print the original data
	logger.debug(json.dumps(serp_data, indent=2))

	# to string
	serp_data_str = json.dumps(serp_data)

	return ActionResult(extracted_content=serp_data_str, include_in_memory=False)


names = [
	'Ruedi Aebersold',
	'Bernd Bodenmiller',
	'Eugene Demler',
	'Erich Fischer',
	'Pietro Gambardella',
	'Matthias Huss',
	'Reto Knutti',
	'Maksym Kovalenko',
	'Antonio Lanzavecchia',
	'Maria Lukatskaya',
	'Jochen Markard',
	'Javier Pérez-Ramírez',
	'Federica Sallusto',
	'Gisbert Schneider',
	'Sonia I. Seneviratne',
	'Michael Siegrist',
	'Johan Six',
	'Tanja Stadler',
	'Shinichi Sunagawa',
	'Michael Bruce Zimmermann',
]


async def main():
	task = 'use search_web with "find email address of the following ETH professor:" for each of the following persons in a list of actions. Finally return the list with name and email if provided - do always 5 at once'
	task += '\n' + '\n'.join(names)
	model = ChatOpenAI(model='gpt-4o')
	browser_profile = BrowserProfile()
	agent = Agent(task=task, llm=model, controller=controller, browser_profile=browser_profile)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: PersonList = PersonList.model_validate_json(result)

		for person in parsed.people:
			print(f'{person.name} - {person.email}')
	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())
