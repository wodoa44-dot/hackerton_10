"""Independently recompute the reported K curves and six slots from exported rows."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'artifacts/encoder_comparison'
SLOTS=['classification','trait','part_number','case','packing','vendor']


def read(name):
    with (DIRECTORY/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def main():
    result=json.loads((DIRECTORY/'comparison.json').read_text(encoding='utf-8'))
    paired=read('paired_predictions.csv')
    assert len(paired)==2850 and len({r['source_row'] for r in paired})==2850
    groups={'all':paired,'manual':[r for r in paired if r['manual_parseable']=='True'],
            'unmapped':[r for r in paired if r['manual_parseable']!='True']}
    assert {key:len(value) for key,value in groups.items()}=={'all':2850,'manual':473,'unmapped':2377}
    checks=0
    for variant,metrics in result['variants'].items():
        candidates=defaultdict(list)
        for row in read(variant+'/full_code_candidates_top10.csv'):
            candidates[row['source_row']].append(row)
        for rows in candidates.values():
            assert len(rows)<=10 and len({r['code'] for r in rows})==len(rows)
            assert [int(r['rank']) for r in rows]==list(range(1,len(rows)+1))
        for scope,rows in groups.items():
            for k in range(1,11):
                count=sum(row['truth'] in [candidate['code'] for candidate in candidates[row['source_row']][:k]] for row in rows)
                point=metrics['candidate_k_metrics']['groups'][scope]['points'][k-1]
                assert point['k']==k and point['n']==len(rows) and point['correct']==count
                assert abs(point['rate']-count/len(rows))<1e-12
                checks+=1
        for slot in SLOTS:
            rows=read(variant+'/'+slot+'_predictions.csv')
            assert len(rows)==2850
            eligible=[row for row in rows if row['eligible']=='True']
            m=metrics['slots'][slot]
            assert m['n']==len(eligible)
            correct=sum(row['truth']==row['displayed_value'] for row in eligible)
            assert abs(m['accuracy']-correct/len(eligible))<1e-12
            for k in [1,3,5,10]:
                point=m['joint_at_k'][str(k)]
                assert point['n']==len(eligible)
                assert point['correct']==sum(row['joint_hit_at_'+str(k)]=='True' for row in eligible)
            checks+=5
        assert metrics['normalized_input_overlap_in_registry']==0
        assert metrics['operational_master_used'] is False
    assert result['protocol']['training_or_selection_uses_evaluation'] is False
    proof={'passed':True,'evaluation_rows':2850,'manual_rows':473,'checks':checks,
           'k_values':list(range(1,11)),'variants':list(result['variants']),
           'scope':'Recomputed complete-code curves and six slot results from exported per-row predictions.'}
    (ROOT/'recovery/encoder_results_verified.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof))


if __name__=='__main__':main()
