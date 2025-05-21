import asyncio
import logging
from typing import Set, Dict, Any, Optional

# 假设以下模块已经实现并导入
from agent.ai_client import AIClient              # 调用 AI 生成备注
from controller.controller import Controller      # 浏览器控制器
from dom.dom_manager import DOMManager             # DOM 快照管理
from telemetry.telemetry_manager import TelemetryManager  # 页面数据采集管理

# 任务管理、页面状态、元素管理、操作树、文档生成和可视化
from player.task_manager import TaskManager
from player.page_state import PageStateManager
from player.element_manager import ElementManager
from player.operation_tree import OperationTreeManager
from player.document_generator import DocumentGenerator
from player.visualization import VisualizationManager
from player.result_manager import ResultManager


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Player:
    def __init__(self, start_url: str):
        self.start_url = start_url

        # 组件实例化
        self.controller = Controller()
        self.ai_client = AIClient()
        self.dom_manager = DOMManager()
        self.telemetry_manager = TelemetryManager()

        self.task_manager = TaskManager()
        self.page_state_manager = PageStateManager()
        self.element_manager = ElementManager()
        self.operation_tree_manager = OperationTreeManager()
        self.document_generator = DocumentGenerator()
        self.visualization_manager = VisualizationManager()
        self.result_manager = ResultManager()

        # 访问历史，防止重复访问同一页面状态（可根据实际设计调整）
        self.visited_pages: Set[str] = set()

    async def run(self):
        logger.info(f"启动文档生成任务，入口地址：{self.start_url}")

        # 初始化根页面任务节点
        root_page_state = await self.load_page(self.start_url)
        self.operation_tree_manager.init_root_node(root_page_state)

        # 启动递归探索
        await self.explore_page(root_page_state)

        # 所有操作完成，生成文档和可视化结果
        await self.finalize()

        logger.info("任务完成。")

    async def load_page(self, url: str) -> Dict[str, Any]:
        """
        加载页面，采集内容和DOM快照，生成页面状态对象。
        """
        # 通过 controller 打开页面
        logger.info(f"打开页面: {url}")
        result = await self.controller.act({"action": "go_to_url", "params": {"url": url}})
        if not result.success:
            logger.error(f"无法打开页面 {url}: {result.message}")
            raise RuntimeError(f"打开页面失败: {url}")

        # 等待页面稳定（可根据需要调整等待时间）
        await asyncio.sleep(2)

        # 采集 DOM 和页面数据
        dom_snapshot = await self.dom_manager.capture_dom_snapshot()
        telemetry_data = await self.telemetry_manager.capture()

        # 提取页面内容摘要（假设 controller 有 extract_content 动作）
        extract_result = await self.controller.act({"action": "extract_content", "params": {"target": "summary"}})
        page_summary = extract_result.data if extract_result.success else ""

        # 构造页面状态字典
        page_state = {
            "url": url,
            "dom_snapshot": dom_snapshot,
            "telemetry": telemetry_data,
            "summary": page_summary,
        }
        logger.debug(f"页面状态构建完成: {url}")
        return page_state

    async def explore_page(self, page_state: Dict[str, Any]):
        """
        递归探索页面，构建操作树。
        """
        url = page_state["url"]

        # 简单去重判定
        if url in self.visited_pages:
            logger.info(f"页面已访问，跳过：{url}")
            return
        self.visited_pages.add(url)

        logger.info(f"探索页面：{url}")

        # 采集所有可点击元素
        elements = await self.element_manager.collect_clickable_elements()
        logger.info(f"发现 {len(elements)} 个可点击元素")

        # AI生成每个元素的备注信息
        for elem in elements:
            elem_desc = await self.ai_client.generate_element_note(page_state["summary"], elem)
            elem["note"] = elem_desc

        # 更新当前节点的操作元素
        self.operation_tree_manager.update_node_elements(url, elements)

        # 对所有可点击元素递归执行
        for elem in elements:
            selector = elem.get("selector")
            if not selector:
                logger.warning("元素缺少选择器，跳过")
                continue

            logger.info(f"点击元素，选择器: {selector}")

            # 调用 controller 执行点击动作
            click_result = await self.controller.act({
                "action": "click_element_by_selector",
                "params": {"selector": selector}
            })
            if not click_result.success:
                logger.warning(f"元素点击失败: {selector}, 继续下一个")
                continue

            # 等待页面响应和加载
            await asyncio.sleep(2)

            # 采集新页面状态
            new_page_state = await self.capture_current_page_state()

            # 添加子节点并递归
            self.operation_tree_manager.add_child_node(parent_url=url, page_state=new_page_state)
            await self.explore_page(new_page_state)

            # 返回原页面（假设 controller 有返回操作）
            back_result = await self.controller.act({"action": "go_back"})
            if not back_result.success:
                logger.error("返回上一页面失败，任务异常终止")
                return
            await asyncio.sleep(1)

    async def capture_current_page_state(self) -> Dict[str, Any]:
        """
        采集当前页面的状态信息（DOM，内容等）
        """
        # 获取当前页面URL，假设 controller 有 get_current_url 动作
        url_result = await self.controller.act({"action": "get_current_url"})
        current_url = url_result.data if url_result.success else "unknown"

        dom_snapshot = await self.dom_manager.capture_dom_snapshot()
        telemetry_data = await self.telemetry_manager.capture()
        extract_result = await self.controller.act({"action": "extract_content", "params": {"target": "summary"}})
        page_summary = extract_result.data if extract_result.success else ""

        page_state = {
            "url": current_url,
            "dom_snapshot": dom_snapshot,
            "telemetry": telemetry_data,
            "summary": page_summary,
        }
        return page_state

    async def finalize(self):
        """
        所有页面探索完成后，调用文档生成和可视化输出接口。
        """
        logger.info("开始生成用户操作文档和流程图")

        # 生成文档内容
        await self.document_generator.generate(self.operation_tree_manager.get_tree())

        # 生成流程图数据
        await self.visualization_manager.generate(self.operation_tree_manager.get_tree())

        # 保存结果
        await self.result_manager.save_all()

        logger.info("文档和流程图生成完成")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python player.py <start_url>")
        sys.exit(1)

    start_url = sys.argv[1]

    player = Player(start_url)
    asyncio.run(player.run())
