"""Actual code-field models. No D/OTHER estimator or family labels as classification targets."""
import json
from pathlib import Path
import joblib
from erp_model.shared.schema import SLOTS
from erp_model.prediction.inference import Predictor

class VerifiedSlotPredictor(Predictor):
    def __init__(self,directory='artifacts/verified_v3',slot=None):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        self.shared=joblib.load(self.directory/'preprocessor.joblib')
        self.slots=[slot] if slot else SLOTS
        self.models={};self.load_errors={}
        for s in self.slots:
            try:self.models[s]=joblib.load(self.directory/(s+'.joblib'))
            except Exception as exc:self.load_errors[s]=type(exc).__name__+': '+str(exc)
        for model in self.models.values():model.feature.text=self.shared['text']
