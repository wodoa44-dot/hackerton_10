import csv,io,json,os
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get("K_URL","https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/")
OUT=os.environ.get("K_PROOF","exploratory-k-browser")


def main():
    errors=[];proof={"url":URL}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1440,"height":1100},permissions=["clipboard-read","clipboard-write"],accept_downloads=True)
        page=context.new_page();page.on("pageerror",lambda e:errors.append(str(e)))
        page.goto(URL,wait_until="networkidle",timeout=60000)
        page.wait_for_function("document.getElementById('connection').textContent.includes('전체 코드')")
        page.locator('[data-sample="thermistor"]').click()
        page.locator("#candidate-limit").select_option("10");page.locator("#run-button").click()
        page.wait_for_function("document.getElementById('counter').textContent === '6 / 6 완료'")
        page.wait_for_function("!document.getElementById('run-button').disabled")
        assert page.locator("#exploratory-panel").is_visible()
        assert page.locator("#exploratory-body tr").count()==10
        assert page.locator("#exploratory-heads tr").count()==6
        assert "S" not in page.locator("#exploratory-heads tr").nth(1).locator("code").all_text_contents()
        code=page.locator("#exploratory-body tr").first.locator("td").nth(1).inner_text()
        page.locator("[data-copy-estimate]").first.click()
        assert page.evaluate("navigator.clipboard.readText()")==code
        proof["unverified_codes_and_six_reference_heads_visible"]=True
        proof["copy_estimate_verified"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-estimates.png")),full_page=True)
        page.locator('.sidebar [data-page="evaluation"]').click()
        page.wait_for_selector("#k-chart svg",state="visible")
        assert page.locator("#k-chart [data-curve]").count()==3
        assert page.locator("#k-chart [data-k]").count()==30
        page.locator('#k-chart [data-series="all"][data-k="10"]').click()
        detail=page.locator("#k-chart-detail").inner_text()
        assert "K=10" in detail and "12.25%" in detail and "68.50%" in detail and "1.05%" in detail,detail
        proof["k10_readout"]=detail
        page.locator('[data-k-series="unmapped"]').click()
        assert page.locator("#k-chart [data-curve]").count()==2
        page.locator('[data-k-series="unmapped"]').click()
        assert page.locator("#k-chart [data-curve]").count()==3
        with page.expect_download() as svg_info:page.locator("#download-k-svg").click()
        svg=Path(svg_info.value.path()).read_text()
        assert "<svg" in svg and "정답 포함률" in svg
        page.locator(".k-exact-values summary").click()
        with page.expect_download() as csv_info:page.locator('a[href="downloads/candidate_k_curve.csv"]').click()
        data=Path(csv_info.value.path()).read_bytes()
        rows=list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        assert len(rows)==30
        proof["legend_and_svg_csv_exports_verified"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-chart.png")),full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.locator("#k-chart svg").evaluate("(el)=>el.getBoundingClientRect().width")<=390
        proof["mobile_no_horizontal_overflow"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-mobile.png")),full_page=True)
        assert not errors,errors;proof["javascript_errors"]=errors
        (ROOT/"artifacts"/(OUT+".json")).write_text(json.dumps(proof,ensure_ascii=False,indent=2))
        print(json.dumps(proof,ensure_ascii=False));browser.close()


if __name__=="__main__":main()
