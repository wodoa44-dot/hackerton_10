"""Exercise real CSV uploads, progress, downloads, evaluation scope, and mobile UI."""
import csv,io,json,os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get("CSV_URL","https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/")
OUT=os.environ.get("CSV_PROOF","csv-audit-browser")


def main():
    errors=[];proof={"url":URL}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1440,"height":1000},accept_downloads=True)
        page=context.new_page();page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(URL,wait_until="networkidle",timeout=60000)
        page.wait_for_selector("#batch-panel",state="visible")
        page.locator("#batch-details summary").click()
        info=page.request.get(URL+"model-info").json()
        assert info["web_lookup_audit"]["evaluation_input_overlap"]==0
        assert info["web_lookup_audit"]["evaluation_code_overlap"]==0
        assert info["web_lookup_audit"]["current_lookup_rows"]==10834
        proof["web_lookup_training_only"]=True
        def upload(text,codec,filename):
            page.locator("#batch-file").set_input_files({"name":filename,"mimeType":"text/csv","buffer":text.encode(codec)})
            page.locator("#batch-upload").click()
            page.wait_for_function("document.getElementById('batch-status').textContent.startsWith('완료') || !document.getElementById('batch-error').hidden",timeout=90000)
            assert page.locator("#batch-error").is_hidden(),page.locator("#batch-error").inner_text()
            with page.expect_download() as download_info:page.locator("#batch-download").click()
            data=Path(download_info.value.path()).read_bytes()
            assert data.startswith(b"\xef\xbb\xbf")
            rows=list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
            return rows
        text="품목명,규격명,품목코드,메모\n기존품목,사양,000123,유지\nRES CHIP,1608 F 12.3K REEL YAGEO,,신규\nCAP CHIP,1608 X7R 0.47uF K 50V REEL,,신규\n,규격,,빈이름\n"
        rows=upload(text,"utf-8-sig","빈코드_UTF8.csv")
        assert rows[0]["품목코드"]=="000123" and rows[0]["메모"]=="유지"
        assert rows[1]["품목코드"]=="RCP10F1232TPY",rows[1]
        assert rows[2]["품목코드"]=="CC10B474K1HR-",rows[2]
        assert rows[1]["예측_세분류품번"]=="1232" and rows[1]["검토필요"]=="예"
        assert rows[3]["품목코드"]=="" and rows[3]["자동처리상태"]=="미생성"
        proof["utf8_existing_codes_preserved_and_blanks_filled"]=True
        proof["six_part_columns"]=all(("예측_"+x) in rows[1] for x in ["분류","특성","세분류품번","형상","Packing","Vendor"])
        proof["progress_summary"]=page.locator("#batch-status").inner_text()
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-desktop.png")),full_page=True)
        rows=upload("품목명,규격명\nRES CHIP,1608 F 12.3K REEL YAGEO\n","cp949","한글_CP949.csv")
        assert rows[0]["품목코드"]=="RCP10F1232TPY"
        proof["cp949_and_missing_code_column"]=True
        page.locator('.sidebar [data-page="evaluation"]').click()
        body=page.locator("#evaluation-body").inner_text()
        assert "55.22%" in body and "91.61%" in body and "465 / 2,850" in body,body
        assert "미지정" in page.locator("#vendor-diagnosis").inner_text()
        assert "상세 코드 채점 가능 범위" in page.locator("#metric-cards").inner_text()
        proof["evaluation_scope_and_designated_vendor_accuracy_visible"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-evaluation.png")),full_page=True)
        page.locator('.sidebar [data-page="workspace"]').click()
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        proof["mobile_no_horizontal_overflow"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-mobile.png")),full_page=True)
        page.locator("#batch-file").set_input_files({"name":"bad.csv","mimeType":"text/csv","buffer":b"code,spec\n,12K\n"})
        page.locator("#batch-upload").click()
        page.wait_for_selector("#batch-error",state="visible")
        assert "품목명" in page.locator("#batch-error").inner_text()
        proof["malformed_csv_error_visible"]=True
        assert not errors,errors
        proof["javascript_errors"]=errors
        (ROOT/"artifacts"/(OUT+".json")).write_text(json.dumps(proof,ensure_ascii=False,indent=2))
        print(json.dumps(proof,ensure_ascii=False))
        browser.close()


if __name__=="__main__":main()
