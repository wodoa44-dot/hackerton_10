"""Run the recovered contracts that do not require the lost trained weights."""
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
CASES=[
 'test_generation.CompanyLayoutTests',
 'test_manual.ManualRulesTests',
 'test_pipeline.ContractTests',
 'test_training_quality.TrainingQualityTests',
 'test_encoder.EncoderTests',
 'test_csv_and_audit.CsvTests.test_encodings_and_optional_code_column',
 'test_csv_and_audit.CsvTests.test_quoted_commas_and_blank_spec',
 'test_csv_and_audit.CsvTests.test_bad_headers_and_malformed_rows_rejected',
 'test_csv_and_audit.CsvTests.test_only_blank_codes_filled_and_original_values_preserved',
 'test_csv_and_audit.CsvTests.test_output_metadata_never_overwrites_input_columns',
 'test_csv_and_audit.EvaluationAuditTests.test_selected_unmapped_field_does_not_get_credit_from_unused_head',
 'test_csv_and_audit.EvaluationAuditTests.test_majority_and_placeholder_metrics_keep_separate_denominators',
 'test_exploratory.ExploratoryTests.test_pattern_adaptation_uses_only_observed_input_span_and_width',
 'test_exploratory.ExploratoryTests.test_top_k_accepts_correct_second_candidate_and_deduplicates',
 'test_exploratory.ExploratoryTests.test_csv_uses_exploratory_code_when_strict_assembly_is_empty',
]

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromNames(CASES)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'skipped':len(result.skipped),'passed':result.wasSuccessful(),
            'scope':'Recovered code contracts and synthetic encoder forward/backward; no pretrained download or model fitting.',
            'lost_weight_dependent_evaluation_executed':False}
    (ROOT/'recovery/source_tests.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
