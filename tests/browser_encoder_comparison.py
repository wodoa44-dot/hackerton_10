"""Validate measured A/B graphs, denominators, K interaction and CSV exports."""
import argparse
import csv
import io
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
        parts={};heads={}
        for i,slot in enumerate(slots):
            n=20 if i==0 else 10-i
            def points(extra):return {str(k):{'n':n,'correct':min(n,k+offset+extra),'rate':min(n,k+offset+extra)/n} for k in range(1,11)}
            placeholders=min(i,n-1)
            parts[slot]={'n':n,'accuracy':.5,'macro_f1':.3,'joint_at_k':points(4),'marginal_at_k':points(2),'audit':{'coverage':n/20,'placeholder_n':placeholders,'designated_n':n-placeholders,'designated_correct':1,'designated_accuracy':1/(n-placeholders)}}
            heads[slot]={'n':n,'accuracy':points(0)['1']['rate'],'macro_f1':.3,'at_k':points(0)}
        variants[name]={'evaluation_rows':20,'candidate_k_metrics':{'groups':groups},'slots':parts,'raw_heads':heads}
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
        page.locator('#encoder-slot-chart-classification svg').wait_for(state='visible',timeout=60000)
        info=page.evaluate("async()=>await (await fetch('model-info')).json()")
        report=info['encoder_comparison'];slots=list(report['variants']['no_encoder']['slots'])
        assert page.locator('[data-slot-panel]').count()==6
        assert page.locator('[data-slot-chart] svg').count()==6
        assert page.locator('[data-slot-curve]').count()==12
        assert page.locator('[data-slot-chart] circle').count()==120
        assert page.locator('#encoder-slot-mode').input_value()=='raw'
        assert not page.locator('#encoder-whole-results').evaluate('el=>el.open')
        def rate(slot,variant,k,mode='raw'):
            variant=report['variants'][variant]
            point=variant['raw_heads'][slot]['at_k'][str(k)] if mode=='raw' else variant['slots'][slot]['marginal_at_k' if mode=='part' else 'joint_at_k'][str(k)]
            return f"{point['rate']*100:.2f}%"
        for slot in slots:
            assert page.locator('#encoder-slot-table-'+slot+' tr').count()==4
            for variant in ['no_encoder','encoder']:
                actual=page.locator(f'#encoder-slot-scores-{slot} [data-variant="{variant}"] strong').inner_text()
                assert actual==rate(slot,variant,5),(slot,variant,actual)
        page.locator('#encoder-slot-k-classification').select_option('10')
        assert 'Top-10' in page.locator('#encoder-slot-detail-classification').inner_text()
        assert page.locator('#encoder-slot-k-trait').input_value()=='5'
        page.locator('#encoder-slot-chart-vendor [data-slot-k="3"][data-variant="encoder"]').click()
        assert page.locator('#encoder-slot-k-vendor').input_value()=='3'
        assert page.locator('#encoder-slot-k-classification').input_value()=='10'
        for mode in ['part','joint']:
            page.locator('#encoder-slot-mode').select_option(mode)
            assert not page.locator('#encoder-slot-macro-classification').is_visible()
            for variant in ['no_encoder','encoder']:
                assert page.locator(f'#encoder-slot-scores-classification [data-variant="{variant}"] strong').inner_text()==rate('classification',variant,10,mode)
        page.locator('#encoder-slot-mode').select_option('raw')
        assert page.locator('#encoder-slot-macro-classification').is_visible()
        for slot in slots:
            with page.expect_download() as download:page.locator(f'[data-slot-csv="{slot}"]').click()
            rows=list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
            assert len(rows)==20 and {row['slot'] for row in rows}=={slot}
            for row in rows:
                expected=report['variants'][row['variant']]['raw_heads'][slot]['at_k'][row['k']]
                assert row['mode']=='raw' and int(row['n'])==expected['n'] and int(row['correct'])==expected['correct']
                assert abs(float(row['rate'])-expected['rate'])<1e-12
        with page.expect_download() as download:page.locator('[data-slot-svg="classification"]').click()
        svg=Path(download.value.path()).read_text(encoding='utf-8')
        assert '<svg' in svg and '분류' in svg
        page.locator('#encoder-whole-results summary').click()
        page.locator('#encoder-chart svg').wait_for(state='visible')
        assert page.locator('#encoder-k-table tr').count()==4
        assert page.locator('#encoder-chart circle').count()==20
        page.locator('#encoder-k').select_option('10')
        assert 'Top-10' in page.locator('#encoder-chart-detail').inner_text()
        assert page.locator('#encoder-slot-k-classification').input_value()=='10'
        assert page.locator('#encoder-slot-k-trait').input_value()=='5'
        page.locator('#encoder-scope').select_option('manual')
        assert '매뉴얼 해석 가능' in page.locator('#encoder-scope-note').inner_text()
        assert page.locator('#encoder-slot-mode').input_value()=='raw'
        with page.expect_download() as download:page.locator('#encoder-chart-save').click()
        assert '<svg' in Path(download.value.path()).read_text(encoding='utf-8')
        page.locator('#encoder-part-variant').select_option('encoder')
        assert '/encoder/' in page.locator('#encoder-part-downloads a').first.get_attribute('href')
        if not a.fixture:
            with page.expect_download() as download:page.locator('#encoder-part-downloads a').first.click()
            assert Path(download.value.path()).stat().st_size>5
        page.locator('#encoder-scope').select_option('all')
        page.locator('#encoder-whole-results summary').click()
        page.evaluate('window.scrollTo(0,0)')
        page.screenshot(path=str(a.out/'desktop.png'),full_page=False)
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(300)
        page.locator('[data-slot-panel="classification"]').scroll_into_view_if_needed()
        page.screenshot(path=str(a.out/'mobile.png'),full_page=False)
        overflow=page.evaluate('''()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,wide:[...document.querySelectorAll('body *')].filter(el=>el.getBoundingClientRect().right>innerWidth+1 && getComputedStyle(el).display!=='none').slice(0,12).map(el=>({tag:el.tagName,id:el.id,cls:el.className,width:el.getBoundingClientRect().width,right:el.getBoundingClientRect().right}))})''')
        assert overflow['scroll']<=overflow['width'],overflow
        for slot in slots:assert page.locator('#encoder-slot-chart-'+slot+' svg').evaluate('el=>el.getBoundingClientRect().width')<=390
        assert not errors,errors
        proof.update(k_values=[1,3,5,10],whole_code_curves=2,independent_slot_panels=6,slot_curves=12,independent_k_controls=True,three_metric_modes_verified=True,slot_csv_exports_verified=6,exported_slot_metric_checks=120,svg_export_verified=True,row_csv_download_verified=not a.fixture,mobile_no_horizontal_overflow=True,javascript_errors=errors)
        (a.out/'proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(proof,ensure_ascii=False));browser.close()


if __name__=='__main__':main()
