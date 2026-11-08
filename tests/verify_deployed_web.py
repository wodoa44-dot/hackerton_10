"""배포된 웹의 실제 모델·6개 이벤트·CSV·평가 파일을 HTTP로 검증합니다."""
import argparse
import csv
import io
import json
import urllib.error
import urllib.request
import time

def verify(base,expected_source):
    base=base.rstrip("/")
    def request(route,payload=None,csv_bytes=None):
        data=json.dumps(payload).encode() if payload is not None else csv_bytes
        headers={"Content-Type":"application/json"} if payload is not None else {"Content-Type":"text/csv","X-Filename":"deployment_dummy.csv"} if csv_bytes is not None else {}
        with urllib.request.urlopen(urllib.request.Request(base+route,data=data,headers=headers),timeout=120) as response:
            return response.read()
    health=json.loads(request("/health"))
    assert health["version"]=="7.0-char-cnn-augmented",health
    assert health["source_commit"]==expected_source,health
    info=json.loads(request("/model-info"))
    assert info["active_model_variant"]=="encoder"
    assert info["encoder_comparison"]["variants"]["encoder"]["designated"]["raw_heads"]["vendor"]["n"]==67
    assert not info["row_result_downloads"]
    index=request("/").decode()
    assert "칸별 예측 결과" not in index
    assert "reports/encoder-topk/" in index
    payload={"name":"DIODE","spec":"1N4148 CANARY 100V","candidate_limit":5}
    events=[json.loads(line) for line in request("/predict",payload=payload).decode().splitlines() if line]
    expected={"classification","trait","part_number","case","packing","vendor"}
    assert expected.issubset({event.get("slot") for event in events}),events
    assert not any(event.get("status")=="error" for event in events),events
    assembly=next(e for e in events if e.get("slot")=="assembly")
    code=assembly.get("display_code") or assembly.get("predicted_code")
    assert code and "OTHER" not in code,assembly
    assert request("/reports/encoder-topk/figures/raw_designated.png").startswith(b"\x89PNG")
    assert "코드항목" in request("/reports/encoder-topk/topk_summary.csv").decode("utf-8-sig")
    for route in ["/reports/encoder-topk/private/predictions/whole_codes.csv","/comparison-downloads/paired_predictions.csv"]:
        try:
            request(route)
            raise AssertionError("행별 private 결과가 공개됐습니다.")
        except urllib.error.HTTPError as error:
            assert error.code==404
    data="품목명,규격명,품목코드\nDIODE,1N4148 CANARY 100V,\nDEMO_EXISTING,DEMO,KEEP_CANARY_CODE\n".encode("utf-8-sig")
    job=json.loads(request("/batch-csv",csv_bytes=data))
    job_id=job["id"]
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        job=json.loads(request("/batch-csv/"+job_id))
        if job["status"]=="completed":
            break
        if job["status"]=="failed":
            raise RuntimeError(job.get("error"))
        time.sleep(.5)
    else:
        raise RuntimeError("CSV 처리 시간 초과")
    rows=list(csv.DictReader(io.StringIO(request("/batch-csv/"+job_id+"/download").decode("utf-8-sig"))))
    assert len(rows)==2 and rows[0]["품목코드"] and rows[1]["품목코드"]=="KEEP_CANARY_CODE",rows
    return {"passed":True,"version":health["version"],"source_commit":health["source_commit"],"six_model_events":True,"predicted_code":code,"csv_blank_filled":True,"csv_existing_preserved":True,"public_report":True,"private_rows_blocked":True,"csv_job_id":job_id}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",required=True)
    parser.add_argument("--expected-source",required=True)
    parser.add_argument("--out")
    args=parser.parse_args()
    result=verify(args.url,args.expected_source)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text,encoding="utf-8")
    print(text)
