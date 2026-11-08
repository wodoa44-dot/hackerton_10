"""기존 joblib/pickle·외부 호출을 위한 모듈 경로 호환 계층."""
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

ALIASES = {
  "erp_model.accuracy_candidates": "erp_model.encoders.candidates",
  "erp_model.accuracy_runtime": "erp_model.prediction.accuracy_runtime",
  "erp_model.artifact_paths": "erp_model.shared.artifact_paths",
  "erp_model.attribute_models": "erp_model.models.attributes",
  "erp_model.batch_csv": "erp_model.api.batch_csv",
  "erp_model.candidates_v2": "erp_model.encoders.legacy_candidates",
  "erp_model.code_metrics": "erp_model.prediction.code_metrics",
  "erp_model.code_rules": "erp_model.encoders.rules.code_rules",
  "erp_model.company_schema": "erp_model.encoders.rules.company_schema",
  "erp_model.compare_encoders": "erp_model.evaluation.compare_encoders",
  "erp_model.encoder_data": "erp_model.training.data.encoder_data",
  "erp_model.encoder_network": "erp_model.encoders.cnn.network",
  "erp_model.encoder_runtime": "erp_model.encoders.cnn.runtime",
  "erp_model.evaluate_encoder": "erp_model.evaluation.evaluate_encoder",
  "erp_model.evaluate_generation": "erp_model.evaluation.evaluate_generation",
  "erp_model.evaluate_spectrum": "erp_model.evaluation.evaluate_spectrum",
  "erp_model.evaluation_audit": "erp_model.evaluation.audit",
  "erp_model.existing_lookup": "erp_model.models.lookup",
  "erp_model.exploratory": "erp_model.prediction.exploratory",
  "erp_model.family_model": "erp_model.models.family",
  "erp_model.features": "erp_model.encoders.features",
  "erp_model.generation": "erp_model.prediction.generation",
  "erp_model.inference": "erp_model.prediction.inference",
  "erp_model.label_quality": "erp_model.encoders.rules.label_quality",
  "erp_model.manual_audit": "erp_model.training.data.manual_audit",
  "erp_model.material_runtime": "erp_model.prediction.material_runtime",
  "erp_model.material_schema": "erp_model.encoders.rules.material_schema",
  "erp_model.numeric_inputs": "erp_model.encoders.numeric_inputs",
  "erp_model.prepare_encoder_data": "erp_model.training.data.prepare_encoder_data",
  "erp_model.preprocess": "erp_model.encoders.preprocess",
  "erp_model.registry": "erp_model.models.registry",
  "erp_model.schema": "erp_model.shared.schema",
  "erp_model.service": "erp_model.api.service",
  "erp_model.spectrum": "erp_model.prediction.spectrum",
  "erp_model.train": "erp_model.training.train",
  "erp_model.train_accuracy": "erp_model.training.train_accuracy",
  "erp_model.train_attributes": "erp_model.training.train_attributes",
  "erp_model.train_comparison_baseline": "erp_model.training.train_comparison_baseline",
  "erp_model.train_encoder": "erp_model.training.train_encoder",
  "erp_model.train_encoder_augmented": "erp_model.training.train_encoder_augmented",
  "erp_model.train_generation": "erp_model.training.train_generation",
  "erp_model.train_material": "erp_model.training.train_material",
  "erp_model.train_verified": "erp_model.training.train_verified",
  "erp_model.training_audit": "erp_model.training.data.training_audit",
  "erp_model.training_io": "erp_model.training.io",
  "erp_model.validate_accuracy": "erp_model.evaluation.validate_accuracy",
  "erp_model.verified_predictor": "erp_model.prediction.verified_predictor"
}

class _AliasLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, old_name, target, filename):
        super().__init__(old_name, str(filename))
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        # 실제 구현은 정식 기능 폴더에서 한 번만 로드합니다.
        return None

class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = ALIASES.get(fullname)
        if canonical is None:
            return None
        filename = Path(__file__).parent.joinpath(*canonical.split(".")[1:]).with_suffix(".py")
        loader = _AliasLoader(fullname, canonical, filename)
        return importlib.util.spec_from_loader(fullname, loader)

def install():
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
