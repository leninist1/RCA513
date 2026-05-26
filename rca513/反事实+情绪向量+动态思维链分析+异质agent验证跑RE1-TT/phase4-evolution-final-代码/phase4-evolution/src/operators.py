from typing import Dict, Any

class OperatorConfig:
    """算子配置容器，您的 RCA 算法根据它来调整行为"""
    def __init__(self, op_id: str, name: str, params: Dict = None):
        self.id = op_id
        self.name = name
        self.params = params or {}

    def to_dict(self):
        return {"id": self.id, "name": self.name, "params": self.params}

def load_operators_from_yaml(filepath: str) -> list:
    import yaml
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    ops = []
    for op in config['operators']:
        ops.append(OperatorConfig(str(op['id']), op['name'], op.get('params', {})))
    return ops