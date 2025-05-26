import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'  # 例如 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
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
		2. 输入法人账号
		3. 输入通用验证码114477
		4. 新建一个验收
		5. 选择验收阶段  
		6. 输入阶段名称
		7. 选择验收时间
		8. 上传验收申请报告两份
		9. 上传资料
		10.上传审核材料（5+1份）
		11.上传参考文件（必要的2份）
		""",
		llm=llm,
		max_actions_per_step=4,
		browser_session=browser_session,
	)

	await agent.run(max_steps=25)


if __name__ == '__main__':
	asyncio.run(run_search())
