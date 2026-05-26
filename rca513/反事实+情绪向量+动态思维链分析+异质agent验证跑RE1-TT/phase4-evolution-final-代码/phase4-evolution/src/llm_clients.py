import os
import abc
import json
import requests
import anthropic

class BaseLLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        pass


class ClaudeClient(BaseLLMClient):
    """Anthropic 原生接口（中转站用自定义 base_url）"""
    def __init__(self, model: str, api_key_env: str, base_url_env: str = None):
        self.model = model
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {api_key_env} 未设置")
        
        # base_url 直接用环境变量的值，不要额外拼接
        if base_url_env:
            base_url = os.environ.get(base_url_env, "")
        else:
            base_url = "https://api.anthropic.com"
        
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API（OpenAI 兼容接口）"""
    def __init__(self, model: str, api_key_env: str, base_url: str = "https://api.deepseek.com"):
        self.model = model
        self.base_url = base_url
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"环境变量 {api_key_env} 未设置")
        self.api_key = api_key

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096
            },
            headers=headers
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]


class MockLLMClient(BaseLLMClient):
    """模拟客户端，测试用"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if "claude" in self.agent_id:
            return json.dumps({
                "variant_id": f"V7.1-{self.agent_id}",
                "parent_version": "V7.0",
                "target_operator": "1",
                "modification": {
                    "what_changes": "调整时间权重",
                    "new_params": {"weight_time": 0.5}
                }
            })
        else:
            return json.dumps({
                "variant_id": f"V7.1-{self.agent_id}",
                "parent_version": "V7.0",
                "target_operator": "1",
                "modification": {
                    "what_changes": "引入新特征",
                    "new_params": {"use_fingerprint": True}
                }
            })


def create_llm_client(agent_cfg: dict, mock: bool = False):
    if mock:
        return MockLLMClient(agent_cfg.get('llm_client', 'mock'))
    
    client_type = agent_cfg.get('llm_client', '')
    model = agent_cfg.get('model', '')
    api_key_env = agent_cfg.get('api_key_env', '')
    base_url_env = agent_cfg.get('base_url_env', None)
    
    if client_type == 'claude':
        return ClaudeClient(model, api_key_env, base_url_env)
    elif client_type == 'deepseek':
        return DeepSeekClient(model, api_key_env)
    else:
        raise ValueError(f"不支持的 LLM 类型: {client_type}")
