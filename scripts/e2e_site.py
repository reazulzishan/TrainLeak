"""End-to-end test of the web app in a real (headless) browser. usage: e2e_site.py URL [train test]"""
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

url = sys.argv[1]
shots = Path("docs/e2e"); shots.mkdir(parents=True, exist_ok=True)
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.goto(url)
    expect(pg.locator("#status")).to_contain_text("Ready", timeout=180_000)
    print("ready:", pg.locator("#status").inner_text())
    pg.screenshot(path=str(shots / "1_ready.png"))

    pg.click("#ex-demo")
    expect(pg.locator("#status")).to_contain_text("Done", timeout=60_000)
    assert "3 leaks found" in pg.locator("#result").inner_text(), pg.locator("#result").inner_text()[:300]
    print("demo:", pg.locator("#status").inner_text())
    pg.screenshot(path=str(shots / "2_demo.png"), full_page=True)

    pg.click("#ex-nb")
    expect(pg.locator("#result")).to_contain_text("SMOTE runs before the train/test split", timeout=60_000)
    pg.click("#ex-fixed")
    expect(pg.locator("#result")).to_contain_text("No leaks found", timeout=60_000)
    print("notebooks: ok")

    if len(sys.argv) == 4:
        pg.click("#tab-split")
        pg.set_input_files("#f1", sys.argv[2]); pg.set_input_files("#f2", sys.argv[3])
        expect(pg.locator("#target")).to_be_enabled(timeout=60_000)
        pg.select_option("#target", "label")
        pg.click("#run")
        expect(pg.locator("#result")).to_contain_text("TL102", timeout=120_000)
        print("upload:", pg.locator("#status").inner_text())
        pg.screenshot(path=str(shots / "3_upload.png"), full_page=True)
    pg.goto(url + "?theme=light")
    expect(pg.locator("#status")).to_contain_text("Ready", timeout=180_000)
    assert pg.evaluate("document.documentElement.dataset.theme") == "light"
    pg.click("#ex-demo"); expect(pg.locator("#result")).to_contain_text("3 leaks found", timeout=60_000)
    assert "#FFFFFF" in pg.locator("#result").inner_html().upper()
    pg.screenshot(path=str(shots / "4_light.png"), full_page=True)
    print("light theme: ok")
    assert not errors, errors
    b.close()
print("E2E PASSED")
