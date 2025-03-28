from typing import List, Optional, Dict, Any

class Solution:
    def __init__(self, title: str, subtitle: str):
        self.title = title
        self.subtitle = subtitle
        self.insights = []

class Insight:
    def __init__(self, text: str, sources: List[str], vega_lite_spec: Optional[Dict[str, Any]] = None):
        self.text = text
        self.sources = sources
        self.vega_lite_spec = vega_lite_spec

class Conversation:
    def __init__(self, problem_text: str, solutions: List[Solution], insights: List[Insight], background_info: dict, chat_history: Optional[List[Dict[str, str]]] = None):
        self.problem_text = problem_text
        self.solutions = solutions
        self.insights = insights
        self.background_info = background_info
        self.chat_history = chat_history or []

    def add_solution(self, solution: Solution):
        self.solutions.append(solution)

    def add_insight(self, insight: Insight):
        self.insights.append(insight)

    def add_message(self, role: str, content: str):
        """
        Add a message to the chat history.

        Args:
            role (str): The role of the message sender ('user' or 'assistant')
            content (str): The content of the message
        """
        self.chat_history.append({"role": role, "content": content}) 