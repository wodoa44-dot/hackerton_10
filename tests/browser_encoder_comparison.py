"""Validate measured A/B graphs, denominators, K interaction and CSV exports."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def fixture_info():
    variants={}
    slots=['classification','trait','part_number','case','packing','vendor']
    for name,offset in [('no_encoder',0),('encoder',1)]:
        groups={}
        for scope,n in [('all',20),('manual',10),('unmapped',10)]:
            points=[{'k':k,'n':n,'correct':min(n,k+offset),'rate':min(n,k+offset)/n,'average_candidates':k} for k in range(1,11)]
            groups[scope]={'n':n,'predicted_rows':n,'points':points}
        part={'n':10,'accuracy':.5+offset*.1,'macro_f1':.3,'joint_at_k':{str(k):{'n':10,'correct':min(10,k+offset),'rate':min(10,k+offset)/10} for k in range(1,11)},'audit':{'coverage':.5,'designated_n':8,'designated_correct':4,'designated_accuracy':.5}}
        variants[name]={'evaluation_rows':20,'candidate_k_metrics':{'groups':groups},'slots':{s:part for s in slots},'raw_heads':{s:{**part,'at_k':part['joint_at_k']} for s in slots}}
    return {'encoder_comparison':{'variants':variants,'protocol':{'fixed_denominators':{'all':20,'manual':10,'unmapped':10}}},'active_model_variant':'no_encoder','model_variants':{key:{'available':True} for key in variants},'schemes':[],'csv_upload':True,'training_counts':{s:{'rows':10,'classes':2} for s in slots},'selection':{},'data_audit':{'train_original':25,'excluded_overlap':5,'company_layout_training_rows':10}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--url',default='https://devopen-gpu.jntp-ax.or.kr/i/OsAdz8rQR7JSXeW6/')
    ap.add_argument('--fixture',action='store_true');ap.add_argument('--out',type=Path,default=ROOT.parent/'tmp/encoder_browser');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    errors=[];proof={'url':a.url,'synthetic_fixture':a.fixture}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='msedge',headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
        if a.fixture:
            a.url='https://erp-comparison.test/'
            info=fixture_info()
            def respond(route):
                request_path=route.request.url.split('erp-comparison.test/')[-1].split('?')[0]
                if request_path=='model-info':return route.fulfill(json=info)
                if request_path.startswith('comparison-downloads/'):
                    return route.fulfill(body='k,correct,n\n1,1,20\n',headers={'Content-Type':'text/csv; charset=utf-8','Content-Disposition':'attachment; filename="comparison.csv"'})
                file=ROOT/'erp_model/web'/(request_path or 'index.html')
                return route.fulfill(path=str(file)) if file.is_file() else route.fulfill(status=404,body='not found')
            context.route('https://erp-comparison.test/**',respond)
        page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(a.url,wait_until='networkidle',timeout=60000)
        page.locator('.sidebar [data-page="evaluation"]').click()
        page.locator('#encoder-chart svg').wait_for(state='visible',timeout=60000)
        assert page.locator('#encoder-slot-table tr').count()==6
        assert page.locator('#encoder-k-table tr').count()==4
        assert page.locator('#encoder-chart circle').count()==20
        page.locator('#encoder-k').select_option('10')
        assert 'Top-10' in page.locator('#encoder-chart-detail').inner_text()
        assert page.locator('#encoder-a-k').inner_text()=='없음 · Top-10'
        page.locator('#encoder-scope').select_option('manual')
        assert '매뉴얼 해석 가능' in page.locator('#encoder-scope-note').inner_text()
        page.locator('#encoder-slot-mode').select_option('raw')
        assert '독립 칸 후보' in page.locator('#encoder-slot-note').inner_text()
        page.locator('#encoder-slot-mode').select_option('final')
        assert '지정값' in page.locator('#encoder-slot-table').inner_text()
        with page.expect_download() as download:page.locator('#encoder-chart-save').click()
        assert '<svg' in Path(download.value.path()).read_text(encoding='utf-8')
        page.locator('#encoder-part-variant').select_option('encoder')
        assert '/encoder/' in page.locator('#encoder-part-downloads a').first.get_attribute('href')
        with page.expect_download() as download:page.locator('#encoder-part-downloads a').first.click()
        assert Path(download.value.path()).stat().st_size>5
        page.locator('#encoder-scope').select_option('all')
        page.screenshot(path=str(a.out/'desktop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert page.locator('#encoder-chart svg').evaluate('el=>el.getBoundingClientRect().width')<=390
        page.screenshot(path=str(a.out/'mobile.png'),full_page=True)
        assert not errors,errors
        proof.update(k_values=[1,3,5,10],curves=2,slots=6,export_verified=True,mobile_no_horizontal_overflow=True,javascript_errors=errors)
        (a.out/'proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(proof,ensure_ascii=False));browser.close()


if __name__=='__main__':main()
