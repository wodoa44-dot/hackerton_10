"""Browser checks for one spectrum, six independent slots, and candidate-coherent parts."""
import json,os
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get("SPECTRUM_URL","https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/")
OUT=os.environ.get("SPECTRUM_PROOF","spectrum-v6-browser")


def main():
    errors=[];proof={"url":URL}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        context=browser.new_context(viewport={"width":1440,"height":1100},permissions=["clipboard-read","clipboard-write"])
        page=context.new_page();page.on("pageerror",lambda error:errors.append(str(error)))
        page.goto(URL,wait_until="networkidle",timeout=60000)
        page.wait_for_function("document.getElementById('connection').textContent.includes('전체 코드')")
        def predict(sample,expected=None):
            page.locator('[data-sample="'+sample+'"]').click();page.locator("#run-button").click()
            page.wait_for_function("document.getElementById('counter').textContent === '6 / 6 완료'",timeout=45000)
            page.wait_for_function("!document.getElementById('run-button').disabled",timeout=45000)
            assert page.locator("#form-error").is_hidden()
            actual=page.locator("#predicted-code").inner_text()
            assert actual!="—"
            if expected:assert actual==expected,(actual,expected)
            assert 0<page.locator("#spectrum-body tr").count()<=5
            assert "OTHER" not in page.locator(".slot-value").all_text_contents()
            parts=page.locator("#parts-body tr td:nth-child(4)").all_text_contents()
            assert "".join(parts)==actual,(parts,actual)
            if page.locator("#code-segments").is_visible():
                assert "".join(page.locator("#code-segments .segment-value").all_text_contents())==actual
            page.locator("#copy-code").click()
            assert page.evaluate("navigator.clipboard.readText()")==actual
            return {"code":actual,"values":page.locator(".slot-value").all_text_contents(),"candidates":page.locator("#spectrum-body tr").count(),"copy_verified":True}
        proof["diode"]=predict("basic","DW1N4148-D3TV")
        page.locator('[data-hypothesis="1"]').click()
        alternative=page.locator("#predicted-code").inner_text()
        assert alternative!=proof["diode"]["code"]
        assert "".join(page.locator("#parts-body tr td:nth-child(4)").all_text_contents())==alternative
        assert "".join(page.locator("#code-segments .segment-value").all_text_contents())==alternative
        page.locator("#copy-code").click();assert page.evaluate("navigator.clipboard.readText()")==alternative
        proof["candidate_selection_updates_parts_and_copy"]=True
        proof["resistor"]=predict("resistor","RCP10F1232TPY")
        proof["capacitor"]=predict("capacitor","CC10B474K1HR-")
        proof["legacy_thermistor"]=predict("thermistor")
        assert proof["legacy_thermistor"]["code"]!="60WH0083541A-R"  # Held-out answer is absent from the lookup index.
        assert page.locator("#card-trait .slot-value").inner_text()=="해석 보류"
        proof["manual_thermistor"]=predict("thermistor_ntc","1775NTX9234B-")
        proof["fuse"]=predict("fuse","25702173A15--")
        proof["varistor"]=predict("varistor")
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-desktop.png")),full_page=True)
        page.locator('[data-sample="resistor"]').click();page.locator("#slot").select_option("part_number");page.locator("#run-button").click()
        page.wait_for_function("document.getElementById('counter').textContent === '1 / 1 완료'")
        page.wait_for_function("!document.getElementById('run-button').disabled")
        assert page.locator("#card-part_number .slot-value").inner_text()=="1232"
        assert page.locator("#card-classification .slot-value").inner_text()=="—"
        assert page.locator("#spectrum-panel").is_hidden();proof["single_slot"]=True
        page.locator('.sidebar [data-page="evaluation"]').click()
        assert "Top 5" in page.locator("#metric-cards").inner_text()
        assert page.locator("#downloads a").count()==4
        page.locator("#downloads a").first.click()
        proof["downloads_visible"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-evaluation.png")),full_page=True)
        page.locator('.sidebar [data-page="workspace"]').click()
        page.set_viewport_size({"width":390,"height":844})
        proof["mobile"]=predict("resistor","RCP10F1232TPY")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        proof["mobile_no_horizontal_overflow"]=True
        page.screenshot(path=str(ROOT/"artifacts"/(OUT+"-mobile.png")),full_page=True)
        assert not errors,errors
        proof["javascript_errors"]=errors
        (ROOT/"artifacts"/(OUT+".json")).write_text(json.dumps(proof,ensure_ascii=False,indent=2))
        print(json.dumps(proof,ensure_ascii=False))
        browser.close()


if __name__=="__main__":main()
