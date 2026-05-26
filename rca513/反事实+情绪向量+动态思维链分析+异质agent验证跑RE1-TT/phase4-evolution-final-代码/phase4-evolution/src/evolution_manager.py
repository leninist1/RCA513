import yaml
import json
import os
from typing import List, Dict
from datetime import datetime
from src.operators import load_operators_from_yaml, OperatorConfig
from src.variant import Variant
from src.evaluator import Evaluator
from src.selector import Selector
from src.internal_state_tracker import InternalStateTracker
from src.failure_memory import FailureMemory
from src.prompt_builder import build_agent_prompt
from src.rca_system import RCASystem
from src.rca_config import RCAConfig
from src.mutation_logger import MutationLogger
from src.llm_clients import create_llm_client
from concurrent.futures import ThreadPoolExecutor, as_completed


class EvolutionManager:
    def __init__(self, config_dir="configs", mock_llm=False):
        with open(os.path.join(config_dir, 'agents.yaml'), 'r') as f:
            self.agents_cfg = yaml.safe_load(f)['agents']
        with open(os.path.join(config_dir, 'evaluation.yaml'), 'r') as f:
            self.eval_cfg = yaml.safe_load(f)
        
        self.base_operators: List[OperatorConfig] = load_operators_from_yaml(
            os.path.join(config_dir, 'operators.yaml'))
        self.base_config = RCAConfig()

        self.llm_clients = {}
        for agent_id, cfg in self.agents_cfg.items():
            self.llm_clients[agent_id] = create_llm_client(cfg, mock=mock_llm)

        self.evaluator = Evaluator(self.eval_cfg)
        self.selector = Selector(k_top=3)
        self.tracker = InternalStateTracker()
        self.failure_memory = FailureMemory()
        self.mutation_logger = MutationLogger("logs")

        self.champion = Variant(
            variant_id="V7.0",
            parent_version="base",
            target_operator="",
            modification={"what_changes": "初始基础版本"},
            proposer_agent="system",
            proposer_internal_state={}
        )
        self.champion.metrics = {'accuracy': 0.448, 'R_total': 0.593}

        self.current_gen = 0
        self.generations = self.eval_cfg.get('generations', 3)
        self.log_dir = "logs"
        self.ckpt_dir = "checkpoints"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def run(self):
        for gen in range(self.generations):
            self.current_gen = gen
            print(f"\n=== 进化第 {gen} 代 ===")
            variants = self._run_one_generation()
            self._log(gen, variants)

    def _call_llm(self, agent_id, cfg):
        """单个agent的LLM调用+评估（线程中执行）"""
        failure_summary = self.failure_memory.summarize(cfg['attribution_philosophy'])
        eval_str = f"总准确率={self.champion.metrics.get('accuracy',0):.2f}"
        prompt = build_agent_prompt(cfg, self.champion, failure_summary, self.current_gen, eval_str)
        client = self.llm_clients[agent_id]
        try:
            raw_response = client.generate(prompt)
            print(f"  [DEBUG] {agent_id} 原始返回 (前200字符): {raw_response[:200]}")
            clean = raw_response.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1]
            elif "```" in clean:
                clean = clean.split("```")[1]
            if "```" in clean:
                clean = clean.split("```")[0]
            start = clean.find('{')
            end = clean.rfind('}')
            if start >= 0 and end > start:
                clean = clean[start:end+1]
            clean = clean.strip()
            proposal = json.loads(clean)
        except Exception as e:
            print(f"  LLM 调用或解析失败 ({agent_id}): {e}")
            return None

        variant = Variant.from_proposal(proposal, agent_id, cfg['internal_state'])
        print(f"  收到变体: {variant.variant_id} (修改算子 {variant.target_operator})")

        metrics, traces, new_config = self._evaluate_variant(variant, cfg['internal_state'])
        variant.metrics = metrics

        if new_config is not None:
            self.mutation_logger.record(
                variant.variant_id, agent_id, self.current_gen,
                self.base_config, new_config, metrics
            )

        new_state = self.tracker.update(agent_id, cfg['internal_state'], variant, metrics, traces)
        cfg['internal_state'] = new_state

        if metrics['accuracy'] < 0.5:
            self.failure_memory.add({
                'philosophy': cfg['attribution_philosophy'],
                'fault_type': 'Type C',
                'reason': f"准确率仅 {metrics['accuracy']:.2f}"
            })

        return variant

    def _run_one_generation(self) -> List[Variant]:
        variants = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(self._call_llm, aid, cfg): aid for aid, cfg in self.agents_cfg.items()}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    variants.append(result)

        parents = self.selector.select(variants, self.champion)
        if parents:
            new_champion = max(parents, key=lambda v: v.metrics.get('R_total', 0))
            self.champion = new_champion
            print(f"\n{'='*60}")
            print(f"🏆 第 {self.current_gen} 代结果")
            print(f"{'='*60}")
            for v in variants:
                m = v.metrics
                print(f"  {v.variant_id} ({v.proposer_agent}):")
                print(f"    Top-1={m.get('accuracy',0):.3f} | Stab={m.get('stability',0):.3f} | Gen={m.get('generalization',0):.3f}")
                print(f"    R_ext={m.get('R_ext',0):.3f} | R_int={m.get('R_int',0):.3f} | R_total={m.get('R_total',0):.3f}")
            print(f"  新 Champion: {self.champion.variant_id}, R_total={self.champion.metrics.get('R_total',0):.3f}")
            print(f"{'='*60}")

        self.mutation_logger.save()
        print(self.mutation_logger.summary())
        return variants

    def _evaluate_variant(self, variant, agent_state):
        print(f"\n  🧪 评估变体: {variant.variant_id}")
        metrics, traces = self.evaluator.evaluate(
            variant, self.champion, agent_state, self.base_operators, self.base_config
        )
        new_config = None
        try:
            from src.rca_config import RCAConfig
            mutations = {}
            if 'new_params' in variant.modification:
                for param, value in variant.modification['new_params'].items():
                    if hasattr(RCAConfig, param):
                        mutations[param] = float(value)
            if mutations:
                new_config = self.base_config.update(mutations)
        except:
            pass
        return metrics, traces, new_config

    def _log(self, gen, variants):
        log_path = os.path.join(self.log_dir, f"gen_{gen:02d}.json")
        log_data = {
            "generation": gen,
            "champion": self.champion.variant_id,
            "variants": [{
                "id": v.variant_id,
                "agent": v.proposer_agent,
                "target": v.target_operator,
                "metrics": v.metrics
            } for v in variants]
        }
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
