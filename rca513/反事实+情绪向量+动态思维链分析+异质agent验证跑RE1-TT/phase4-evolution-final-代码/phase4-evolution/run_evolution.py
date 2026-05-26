import sys
from src.evolution_manager import EvolutionManager

if __name__ == "__main__":
    # 设置 mock=True 可以先用模拟 LLM 跑通流程
    mock = "--mock" in sys.argv
    manager = EvolutionManager(mock_llm=mock)
    manager.run()