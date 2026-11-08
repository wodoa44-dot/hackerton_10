"""모델 구현과 이전 erp_model.models pickle 경로를 함께 제공합니다."""
import importlib
__all__ = ["classifier", "ClassifierBundle", "RankBundle", "WEIGHTS"]
def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(importlib.import_module(".tabular", __name__), name)
    globals()[name] = value
    return value
