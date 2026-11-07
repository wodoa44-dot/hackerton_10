"""CSV blank-code filling using the same frozen predictor, without learning uploads."""
import csv,io,json,threading,time,uuid,unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import HTTPException,Request
from fastapi.responses import FileResponse,Response
from .schema import SLOTS
from .spectrum import describe_code

MAX_BYTES=20*1024*1024
MAX_ROWS=30000
LABELS={"classification":"분류","trait":"특성","part_number":"세분류품번","case":"형상","packing":"Packing","vendor":"Vendor"}


def parse_csv(data):
    if not data or len(data)>MAX_BYTES:raise ValueError("CSV 파일은 20MB 이하로 올려주세요.")
    text=None;encoding=None
    for codec in ["utf-8-sig","cp949"]:
        try:text=data.decode(codec);encoding=codec;break
        except UnicodeDecodeError:pass
    if text is None:raise ValueError("UTF-8 또는 CP949로 저장한 CSV 파일을 올려주세요.")
    if "\0" in text:raise ValueError("CSV 텍스트 파일을 올려주세요. 엑셀 파일은 CSV로 저장하세요.")
    try:dialect=csv.Sniffer().sniff(text[:8192],delimiters=",;\t")
    except csv.Error:dialect=csv.excel
    reader=csv.reader(io.StringIO(text,newline=""),dialect)
    try:headers=next(reader)
    except StopIteration:raise ValueError("CSV에 제목 행이 없습니다.")
    normalize=lambda v:"".join(unicodedata.normalize("NFKC",v).split()).lower()
    normalized=[normalize(h) for h in headers]
    if not headers or any(not h for h in normalized) or len(set(normalized))!=len(headers):
        raise ValueError("비어 있거나 중복된 열 제목이 있습니다.")
    def find(aliases):
        return next((headers[normalized.index(a)] for a in aliases if a in normalized),None)
    name=find(["품목명","item_name","name"])
    spec=find(["규격명","spec","specification"])
    code=find(["품목코드","item_code","code"])
    if not name:raise ValueError("품목명 열이 필요합니다. 규격명과 품목코드 열도 함께 사용할 수 있습니다.")
    original_width=len(headers)
    if not code:code="품목코드";headers.append(code)
    rows=[]
    for line,values in enumerate(reader,2):
        if not values:continue
        if len(values)!=original_width:raise ValueError(str(line)+"행의 열 개수가 제목 행과 다릅니다.")
        if original_width<len(headers):values.append("")
        rows.append(dict(zip(headers,values)))
        if len(rows)>MAX_ROWS:raise ValueError("한 번에 30,000행 이하로 올려주세요.")
    if not rows:raise ValueError("CSV에 처리할 데이터 행이 없습니다.")
    return headers,rows,{"name":name,"spec":spec,"code":code,"encoding":encoding}


def output_columns(headers):
    columns={}
    for key,label in [("status","자동처리상태"),("reason","자동처리근거"),("review","검토필요"),
                      *[(s,"예측_"+LABELS[s]) for s in SLOTS],("parts","예측_세부파트"),("candidates","예측_후보코드"),
                      ("exploratory","규칙미확인_추정후보"),("estimated_slots","규칙미확인_칸별참고후보")]:
        value=label;index=2
        while value in headers:value=label+"_"+str(index);index+=1
        headers.append(value);columns[key]=value
    return columns


class CsvJobs:
    def __init__(self,directory,factory):
        self.root=Path(directory)/"batch_jobs";self.factory=factory;self.jobs={}
        self.lock=threading.RLock();self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="csv-code")
    def close(self):self.executor.shutdown(wait=False)
    def create(self,data,filename="input.csv"):
        headers,rows,mapping=parse_csv(data)
        job_id=uuid.uuid4().hex
        with self.lock:
            if sum(j["status"] in {"queued","running"} for j in self.jobs.values())>=5:
                raise ValueError("대기 중인 파일이 많습니다. 진행 중인 처리가 끝난 뒤 다시 올려주세요.")
            job={"id":job_id,"filename":filename[:200],"status":"queued","total":len(rows),"processed":0,
                 "kept":0,"filled":0,"review":0,"unfilled":0,"errors":0,"preview":[],
                 "created_at":time.time(),"encoding":mapping["encoding"],"error":None}
            self.jobs[job_id]=job
        self.root.mkdir(parents=True,exist_ok=True)
        (self.root/job_id).mkdir()
        self._save(job)
        self.executor.submit(self._run,job_id,headers,rows,mapping)
        return self.get(job_id)
    def _save(self,job):
        path=self.root/job["id"]/"status.json";temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job,ensure_ascii=False),encoding="utf-8");temporary.replace(path)
    def get(self,job_id):
        if len(job_id)!=32 or any(c not in "0123456789abcdef" for c in job_id):raise KeyError(job_id)
        with self.lock:
            if job_id in self.jobs:return json.loads(json.dumps(self.jobs[job_id]))
        path=self.root/job_id/"status.json"
        if not path.exists():raise KeyError(job_id)
        job=json.loads(path.read_text(encoding="utf-8"))
        if job["status"] in {"running","queued"}:
            job.update(status="failed",error="서버가 다시 시작되어 처리가 중단됐습니다. 파일을 다시 올려주세요.")
        return job

    @staticmethod
    def batch_inputs(predictor,chunk,mapping,cache):
        """Batch learned-head scoring while preserving input-only per-row preparation."""
        if not hasattr(predictor,"state") or not hasattr(predictor.state,"model"):return {}
        from .preprocess import preprocess
        from .accuracy_candidates import candidates
        prepared={};score=[]
        for i,row in chunk:
            name=row[mapping["name"]];spec=row.get(mapping["spec"],"") if mapping["spec"] else ""
            if row[mapping["code"]].strip() or not name.strip() or (name,spec) in cache:continue
            p=preprocess(name,spec);prepared[i]=p
            if predictor.state.registry.exact.get(p.key):continue
            p.candidates=candidates(p.normalized);p._identifier_pool_ready=True;score.append(p)
        if not score:return prepared
        for slot in SLOTS:
            cfg=predictor.state.manifest["slots"][slot]
            if cfg.get("model_status")!="trained":continue
            model=predictor.state.model(slot)
            if slot=="part_number":
                for p,ranking in zip(score,model.rankings(score)):p._generation_ranking=ranking["ensemble"]
            else:
                probabilities=model.probabilities(score,cfg["classes"])[cfg["model"]]
                for p,probability in zip(score,probabilities):
                    if not hasattr(p,"_generation_probabilities"):p._generation_probabilities={}
                    p._generation_probabilities[slot]=probability
        return prepared

    def _run(self,job_id,headers,rows,mapping):
        job=self.jobs[job_id];columns=output_columns(headers);cache={}
        try:
            predictor=self.factory();predictor.candidate_limit=10
            job.update(status="running",model_version=predictor.state.manifest["version"],
                       lookup_scope=predictor.state.registry.provenance)
            self._save(job)
            temporary=self.root/job_id/"result.pending";result=self.root/job_id/"result.csv"
            with temporary.open("w",encoding="utf-8-sig",newline="") as output:
                writer=csv.DictWriter(output,fieldnames=headers);writer.writeheader()
                for start in range(0,len(rows),32):
                    chunk=list(enumerate(rows[start:start+32],start))
                    try:prepared=self.batch_inputs(predictor,chunk,mapping,cache)
                    except Exception:prepared={}  # Row-level inference still gives explicit errors.
                    for index,row in chunk:
                        for value in columns.values():row[value]=""
                        code=row[mapping["code"]];name=row[mapping["name"]];spec=row.get(mapping["spec"],"") if mapping["spec"] else ""
                        values={};parts=[];status="";reason="";review=False;candidate_codes=[]
                        try:
                            if code.strip():
                                status="기존코드유지";reason="입력 CSV의 코드";job["kept"]+=1
                                detail=describe_code(predictor.state.book,code.strip().upper())
                                values=detail["values"];parts=detail["parts"]
                            elif not name.strip():
                                status="미생성";reason="품목명이 비어 있습니다.";job["unfilled"]+=1
                            else:
                                key=(name,spec)
                                if key not in cache:
                                    kwargs={"prepared":prepared[index]} if index in prepared else {}
                                    events=list(predictor.events(name,spec,**kwargs))
                                    cache[key]=next(e for e in events if e["slot"]=="assembly")
                                assembly=cache[key];spectrum=assembly.get("spectrum",[])
                                code=assembly.get("display_code") or ""
                                exploration=assembly.get("exploratory",{})
                                extra=exploration.get("candidates",[])
                                row[columns["exploratory"]]=" | ".join(c["code"] for c in extra)
                                row[columns["estimated_slots"]]=json.dumps(exploration.get("head_estimates",{}),ensure_ascii=False) if exploration.get("active") else ""
                                forced=not code and bool(extra)
                                if forced:
                                    chosen=extra[0];code=chosen["code"];spectrum=extra
                                    assembly={**assembly,"auto_confirmed":False,"code_source":chosen["source"],
                                              "predicted_values":chosen["values"],"parts":chosen["parts"]}
                                if code:
                                    row[mapping["code"]]=code;job["filled"]+=1
                                    first=spectrum[0] if spectrum else {}
                                    review=not bool(assembly.get("auto_confirmed"));job["review"]+=review
                                    status="추정채움·규칙미검증" if forced else "예측채움·검토필요" if review else "등록일치채움"
                                    reason=assembly.get("code_source") or "예측 결과"
                                    if first.get("input_numeric_consistent") is False:reason+=" · 입력 수치와 충돌"
                                    values=first.get("values",assembly.get("predicted_values",{}))
                                    parts=first.get("parts",assembly.get("parts",[]));candidate_codes=[c["code"] for c in (extra if exploration.get("active") and extra else spectrum)]
                                else:
                                    status="미생성";reason=assembly.get("reason") or "코드 규칙 또는 입력 근거가 부족합니다.";job["unfilled"]+=1
                        except Exception as error:
                            status="행처리오류";reason=type(error).__name__+": "+str(error);job["errors"]+=1
                        row[columns["status"]]=status;row[columns["reason"]]=reason;row[columns["review"]]="예" if review else ""
                        for slot in SLOTS:row[columns[slot]]=values.get(slot) or ""
                        row[columns["parts"]]=json.dumps(parts,ensure_ascii=False) if parts else ""
                        row[columns["candidates"]]=" | ".join(candidate_codes)
                        writer.writerow(row)
                        with self.lock:
                            job["processed"]+=1
                            if len(job["preview"])<5:job["preview"].append({"row":index+2,"name":name,"code":row[mapping["code"]],"status":status})
                        if job["processed"]%10==0:self._save(job)
            temporary.replace(result)
            with self.lock:job.update(status="completed",finished_at=time.time())
        except Exception as error:
            with self.lock:job.update(status="failed",error=type(error).__name__+": "+str(error))
        finally:self._save(job)


def register_csv_routes(app,directory,factory):
    jobs=CsvJobs(directory,factory);app.state.csv_jobs=jobs
    @app.post("/batch-csv",status_code=202)
    async def upload(request:Request):
        data=bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data)>MAX_BYTES:raise HTTPException(413,"CSV 파일은 20MB 이하로 올려주세요.")
        try:return jobs.create(bytes(data),request.headers.get("X-Filename","input.csv"))
        except (ValueError,csv.Error) as error:raise HTTPException(422,str(error))
    @app.get("/batch-csv/{job_id}")
    def status(job_id:str):
        try:return jobs.get(job_id)
        except KeyError:raise HTTPException(404,"처리 기록이 없습니다.")
    @app.get("/batch-csv/{job_id}/download")
    def download(job_id:str):
        try:job=jobs.get(job_id)
        except KeyError:raise HTTPException(404,"처리 기록이 없습니다.")
        if job["status"]!="completed":raise HTTPException(409,"CSV 처리 완료 후 다운로드할 수 있습니다.")
        return FileResponse(jobs.root/job_id/"result.csv",media_type="text/csv; charset=utf-8",filename="ERP_코드채움.csv")
