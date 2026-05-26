from collections import deque
from typing import Dict

class FailureMemory:
    def __init__(self, capacity=20):
        self.memory = deque(maxlen=capacity)

    def add(self, entry: Dict):
        self.memory.append(entry)

    def summarize(self, philosophy: str) -> str:
        relevant = [e for e in self.memory if e.get('philosophy') == philosophy]
        if not relevant:
            return "暂无相关失败记录。"
        summary = "近期失败教训：\n"
        for i, entry in enumerate(relevant[-3:]):
            summary += f"{i+1}. 故障类型 {entry.get('fault_type','?')}: {entry.get('reason','')}\n"
        return summary