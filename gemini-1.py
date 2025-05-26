import asyncio
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, BrowserConfig
from browser_use.browser import BrowserProfile, BrowserSession

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
		planner_llm=llm,
		browser_session=browser_session,
		max_actions_per_step=4,
		save_conversation_path='/log/agent_conversation',
	)

	await agent.run(max_steps=25)


if __name__ == '__main__':
	asyncio.run(run_search())
