"""Verify the deployed six-slot UI, actual code order, clipboard, and mobile layout."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/"


def main():
    errors, proof = [], {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 1100},
                                      permissions=["clipboard-read", "clipboard-write"])
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_function("document.getElementById('connection').textContent.includes('전체 코드')")

        def predict(expected=None):
            page.locator("#run-button").click()
            page.wait_for_function("document.getElementById('counter').textContent === '6 / 6 완료'", timeout=45000)
            page.wait_for_function("!document.getElementById('run-button').disabled")
            assert page.locator("#form-error").is_hidden()
            actual = page.locator("#predicted-code").inner_text()
            assert actual != "—"
            if expected is not None:
                assert actual == expected, (expected, actual)
            assert "OTHER" not in page.locator(".slot-value").all_text_contents()
            if page.locator("#code-segments").is_visible():
                fragments = page.locator("#code-segments .segment-value").all_text_contents()
                assert "".join(fragments) == actual, (fragments, actual)
                positions = page.locator("[data-code-position]").evaluate_all("(items)=>items.map(x=>Number(x.dataset.codePosition))")
                assert positions == sorted(positions)
            assert page.locator("#copy-code").is_enabled()
            page.locator("#copy-code").click()
            page.wait_for_timeout(100)
            assert page.evaluate("navigator.clipboard.readText()") == actual
            return {"code": actual, "values": page.locator(".slot-value").all_text_contents(), "copy_verified": True}

        def sample(kind, expected=None):
            page.locator('[data-sample="' + kind + '"]').click()
            return predict(expected)

        proof["diode"] = sample("basic", "DW1N4148-D3TV")
        proof["resistor"] = sample("resistor", "RCP10F1232TPY")
        proof["capacitor"] = sample("capacitor", "CC10B474K1HR-")
        proof["legacy_thermistor"] = sample("thermistor", "60WH0083541A-R")
        assert page.locator("#card-classification .slot-value").inner_text() == "60"
        page.locator("#name").fill("CHIP RES")
        page.locator("#spec").fill("2012 F 49.9R RC2012F49R9CS")
        proof["customer_code"] = predict("JSDSMPT0077--")
        assert page.locator("#card-trait .slot-value").inner_text() == "해석 보류"
        proof["manual_thermistor"] = sample("thermistor_ntc", "1775NTX9234B-")
        assert page.locator("#card-packing .slot-value").inner_text() == "B"
        proof["fuse"] = sample("fuse", "25702173A15--")
        assert page.locator("#card-packing .slot-value").inner_text() == "해당 없음"
        proof["varistor"] = sample("varistor")
        assert proof["varistor"]["code"].startswith("259014D681-")
        assert page.locator("#card-case .slot-value").inner_text() == "14D"
        page.screenshot(path=str(ROOT / "artifacts/generation-v5-desktop.png"), full_page=True)

        page.locator('[data-sample="resistor"]').click()
        page.locator("#slot").select_option("part_number")
        page.locator("#run-button").click()
        page.wait_for_function("document.getElementById('counter').textContent === '1 / 1 완료'")
        page.wait_for_function("!document.getElementById('run-button').disabled")
        assert page.locator("#card-part_number .slot-value").inner_text() == "1232"
        assert page.locator("#card-classification .slot-value").inner_text() == "—"
        proof["single_slot"] = True

        page.locator('.sidebar [data-page="evaluation"]').click()
        assert "D 정밀도" not in page.locator("#metric-cards").inner_text()
        assert "실제 코드 분류 정확도" in page.locator("#metric-cards").inner_text()
        assert "매뉴얼·CSV 대조" in page.locator("#actual-code-evaluation").inner_text()
        proof["manual_comparison_visible"] = True
        page.locator('.sidebar [data-page="workspace"]').click()
        page.set_viewport_size({"width": 390, "height": 844})
        proof["mobile_resistor"] = sample("resistor", "RCP10F1232TPY")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        proof["mobile_no_horizontal_overflow"] = True
        page.screenshot(path=str(ROOT / "artifacts/generation-v5-mobile.png"), full_page=True)
        assert not errors, errors
        proof["javascript_errors"], proof["url"] = errors, URL
        (ROOT / "artifacts/generation-v5-browser.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2))
        print(json.dumps(proof, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
