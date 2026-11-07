import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix

from erp_model.label_quality import explicit_resistance, unambiguous_indices
from erp_model.attribute_models import AttributeBundle
from erp_model.company_schema import CompanySchemas
from erp_model.preprocess import preprocess
from erp_model.training_audit import audit_training

ROOT=Path(__file__).resolve().parents[1]

class TrainingQualityTests(unittest.TestCase):
    def test_explicit_resistance_rejects_model_and_power_fragments(self):
        self.assertAlmostEqual(explicit_resistance("CEMENT MPR-5R(S),5W,10MR,5%"), .01)
        self.assertEqual(explicit_resistance("150KR F 1/2"), 150000)
        self.assertAlmostEqual(explicit_resistance("MPR-5R(S)0R005(5mR) J,5W"), .005)
        self.assertIsNone(explicit_resistance("MPR-5R(S),5W"))
        self.assertIsNone(explicit_resistance("1608 J 30"))
        self.assertIsNone(explicit_resistance("10K / 20K"))
        self.assertEqual(explicit_resistance("4K7 5%"),4700)
    def test_single_target_groups_are_kept_and_multi_target_groups_are_masked(self):
        keys=[("A","X"),("A","X"),("B","X"),("B","X")]
        self.assertEqual(unambiguous_indices(keys,["1","2","3","3"]),[2,3])
        # Identical words in different verified schemas are different conditional inputs.
        self.assertEqual(unambiguous_indices([("A","RCP"),("A","CC")],["1","2"]),[0,1])
    def test_attribute_fit_filters_conflicts_before_estimator_fit(self):
        x=csr_matrix(np.ones((4,2),dtype=np.float32))
        with patch("erp_model.attribute_models.classifier") as factory:
            group=AttributeBundle.fit_group(x,["A","B","C","C"],["same","same","clean","clean"])
            factory.assert_not_called()
        self.assertEqual(group["classes"],["C"])
        self.assertEqual(group["rows"],2)
        self.assertEqual(group["excluded_conflicting_input_rows"],2)
        self.assertIsNone(AttributeBundle.fit_group(x[:2],["A","B"],["same","same"]))
    def test_audit_uses_only_training_and_reconciles_masks(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);train=base/"training.csv"
            source=[
                {"품목명":"RES CHIP","규격명":"1608 F 12.3K","품목코드":"RCP10F1232TPY"},
                {"품목명":"UNMAPPED","규격명":"ABC","품목코드":"3612345678"},
                {"품목명":"RES CHIP","규격명":"2012 F 44.2K","품목코드":"RCP21F4222TP-"}]
            with train.open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
            for name in ["company_rules.json","manual_source.json"]:
                (base/name).write_bytes((ROOT/"artifacts/generation_v6"/name).read_bytes())
            with (base/"fold_assignments.csv").open("w",encoding="utf-8-sig",newline="") as f:
                w=csv.writer(f);w.writerow(["source_row","normalized_name","normalized_spec","fold","classification"])
                for i,r in enumerate(source):
                    p=preprocess(r["품목명"],r["규격명"])
                    w.writerow([i+2,*p.key,i,"36" if i==1 else "R"])
            counts={s:{"rows":3 if s=="classification" else 0 if s=="part_number" else 2}
                    for s in ["classification","trait","part_number","case","packing","vendor"]}
            (base/"training_counts.json").write_text(json.dumps(counts))
            result=audit_training(str(train),str(base))
            self.assertTrue(result["training_only"]);self.assertFalse(result["inference_targets_read"])
            self.assertTrue(all(result["reconciles_with_saved_training_counts"].values()))
            self.assertEqual(result["retained_manual_status"],{"matched":2,"outside_manual":1})
            self.assertEqual(result["numeric_review_rows"],1)
            self.assertEqual(result["slots"]["part_number"]["exclusion_counts_retained"]["deterministic_numeric_encoding"],2)
            rows=list(csv.DictReader((base/"training_audit/training_rows.csv").open(encoding="utf-8-sig")))
            self.assertEqual(len(rows),3);self.assertEqual(rows[1]["trait_eligible"],"False")

if __name__=="__main__":unittest.main()
