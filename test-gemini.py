# 安装必要库（先执行）
# pip install google-generativeai

import google.generativeai as genai
import os
import requests  # 新增requests库用于自定义会话
from langchain_google_genai import ChatGoogleGenerativeAI
# ------------------ VPN配置部分 ------------------
# 方案一：通过环境变量设置全局代理（适用于简单场景）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'  # 例如 'http://127.0.0.1:1080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'


# ================= VPN 配置部分 =================
# 请选择其中一种配置方案（推荐方案二）

# ----------- 方案二：自定义会话配置（推荐）-----------
class CustomSession:
    def __init__(self):
        self.session = requests.Session()
        # 代理配置（根据实际情况修改）
        self.session.proxies = {
            'http': 'http://127.0.0.1:7897',  # 替换为你的VPN地址和端口
            'https': 'http://127.0.0.1:7897'  # 支持socks5:// 或 http://
        }
        # 可选高级配置
        self.session.timeout = 15  # 超时时间
        # self.session.verify = False  # 关闭SSL验证（慎用）
        
    def send(self, request, **kwargs):
        return self.session.send(request, **kwargs)

# 初始化SDK配置
genai.configure(
    api_key="AIzaSyDw8wDt0oP7oR_xFaecMyJ-Ub22ETreI7g"  # 替换为你的实际API密钥
)

# ================= 业务逻辑部分 =================
model = genai.GenerativeModel('gemini-2.0-flash-exp')

def basic_chat(prompt):
    """基础对话示例"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API调用出错: {str(e)}"

def stream_chat(prompt):
    """流式响应示例"""
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            yield chunk.text
    except Exception as e:
        yield f"流式请求出错: {str(e)}"
from langchain_google_genai import ChatGoogleGenerativeAI
import os




if __name__ == "__main__":
    # 初始化模型（会自动继承代理配置）
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key="AIzaSyDw8wDt0oP7oR_xFaecMyJ-Ub22ETreI7g"  # 替换为你的API密钥
    )

    # 测试调用
    response = llm.invoke("用一句话解释AI")
    print(response.content)
    # 网络连接测试
    try:
        print("正在测试VPN连接...")
        test_response = basic_chat("回复'OK'")
        print("\nVPN连接测试成功 ✅")
    except Exception as e:
        print("\n连接失败 ❌ 错误信息：")
        print(f"类型：{type(e).__name__}")
        print(f"详情：{str(e)}")
        print("\n排查建议：")
        print("1. 检查代理地址和端口是否正确")
        print("2. 确认VPN服务已启动")
        print("3. 尝试关闭防火墙/杀毒软件")
        exit()

    # 正式问答测试
    question = "用简单的语言解释量子计算"
    print(f"\n问题：{question}")
    print("回答：", basic_chat(question))
    
    # 流式响应测试
    print("\n流式响应演示（AI故事生成）：")
    for text in stream_chat("写一个200字关于人工智能的科幻小故事"):
        print(text, end="", flush=True)