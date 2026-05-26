"""
MutationLogger - 记录每代变体的参数改动
用于 evolutionary analysis：哪些参数被LLM改动最频繁、改动幅度最大
"""
import json
import os
from typing import Dict, List
from src.rca_config import RCAConfig


class MutationLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.mutations: List[Dict] = []  # 所有变体的 mutation 记录
        os.makedirs(log_dir, exist_ok=True)
    
    def record(self, variant_id: str, agent: str, generation: int,
               base_config: RCAConfig, mutated_config: RCAConfig,
               metrics: Dict):
        """记录一次变体的参数改动"""
        diff = mutated_config.diff(base_config)
        
        entry = {
            "generation": generation,
            "variant_id": variant_id,
            "agent": agent,
            "metrics": {
                "accuracy": metrics.get("accuracy", 0),
                "top3_rate": metrics.get("top3_rate", 0),
                "mrr": metrics.get("mrr", 0),
                "R_total": metrics.get("R_total", 0)
            },
            "diff": diff
        }
        self.mutations.append(entry)
    
    def save(self):
        """保存到 JSON 文件"""
        filepath = os.path.join(self.log_dir, "mutations.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.mutations, f, indent=2, ensure_ascii=False)
        print(f"[MutationLogger] 已保存 {len(self.mutations)} 条变异记录到 {filepath}")
    
    def summary(self) -> str:
        """生成参数改动摘要"""
        if not self.mutations:
            return "暂无变异记录"
        
        param_stats = {}
        for entry in self.mutations:
            for param, change in entry["diff"].items():
                if param not in param_stats:
                    param_stats[param] = {"count": 0, "total_delta": 0}
                param_stats[param]["count"] += 1
                param_stats[param]["total_delta"] += abs(change["delta"])
        
        lines = ["\n📊 Mutation 统计:"]
        for param, stats in sorted(param_stats.items(), key=lambda x: x[1]["count"], reverse=True):
            lines.append(f"  {param}: 改动{stats['count']}次, 累计变化{stats['total_delta']:.2f}")
        return "\n".join(lines)
