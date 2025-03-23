from typing import List, Optional, Dict, Any

class Solution:
    def __init__(self, title: str, subtitle: str):
        self.title = title
        self.subtitle = subtitle

class Insight:
    def __init__(self, text: str, sources: List[str], vega_lite_spec: Optional[Dict[str, Any]] = None):
        self.text = text
        self.sources = sources
        self.vega_lite_spec = vega_lite_spec

class Conversation:
    def __init__(self, problem_text: str, solutions: List[Solution], insights: List[Insight]):
        self.problem_text = problem_text
        self.solutions = solutions
        self.insights = insights

    def add_solution(self, solution: Solution):
        self.solutions.append(solution)

    def add_insight(self, insight: Insight):
        self.insights.append(insight) 