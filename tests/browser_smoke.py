"""Real-browser regression for visible predictions at the deployed URL.

Run: .venv/bin/python tests/browser_smoke.py
Requires Playwright Chromium and system libraries.
"""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
URL='https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/'
def main():
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        page.goto(URL,wait_until='networkidle')
        page.locator('[data-sample="basic"]').click()
        page.locator('#run-button').click()
        page.wait_for_function("document.getElementById('counter').textContent === '6 / 6 완료'",timeout=30000)
        page.wait_for_function("!document.getElementById('run-button').disabled")
        values=page.locator('.slot-value').all_text_contents()
        assert len(values)==6 and all(v!='—' for v in values),values
        assert values[2]=='1N4148-',values
        assert page.locator('#card-part_number .slot-value-label').inner_text()=='1순위 예측값'
        assert page.locator('#form-error').is_hidden()
        assert page.locator('#predicted-code').inner_text()==''.join(values)
        assert len(page.locator('#predicted-code').inner_text())==13
        assert page.locator('#copy-code').is_enabled()
        page.locator('#card-part_number .detail-button').click()
        assert page.locator('#card-part_number .details').is_visible()
        page.screenshot(path=str(ROOT/'artifacts/web-results-desktop.png'),full_page=True)
        page.locator('[data-sample="conflict"]').click();page.locator('#run-button').click()
        page.wait_for_function("document.getElementById('counter').textContent === '6 / 6 완료'",timeout=30000)
        page.wait_for_function("!document.getElementById('run-button').disabled")
        assert '여러 제조사' in page.locator('#card-vendor .result-reason').inner_text()
        assert page.locator('#assembly-badge').inner_text()=='예측 완료 · 검토 권장'
        page.locator('#slot').select_option('part_number');page.locator('#run-button').click()
        page.wait_for_function("document.getElementById('counter').textContent === '1 / 1 완료'",timeout=30000)
        page.wait_for_function("!document.getElementById('run-button').disabled")
        assert page.locator('#card-part_number .slot-value').inner_text()!='—'
        assert page.locator('#card-classification .slot-value').inner_text()=='—'
        page.set_viewport_size({'width':390,'height':844})
        page.locator('[data-sample="basic"]').click();page.locator('#run-button').click()
        page.wait_for_function("document.getElementById('counter').textContent === '6 / 6 완료'",timeout=30000)
        page.wait_for_function("!document.getElementById('run-button').disabled")
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(ROOT/'artifacts/web-results-mobile.png'),full_page=True)
        assert not errors,errors
        proof={'url':URL,'browser':'Chromium','basic_visible_values':values,'predicted_code_visible_and_copyable':True,'review_candidate_visible':True,'details_expand':True,'vendor_conflict':True,'single_slot':True,'mobile_no_horizontal_overflow':True,'javascript_errors':errors}
        (ROOT/'artifacts/browser_smoke.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(proof,ensure_ascii=False,indent=2))
        browser.close()
if __name__=='__main__':main()
