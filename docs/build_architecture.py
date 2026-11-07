from pathlib import Path
import json
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Bbox
from PIL import Image

OUT = Path(__file__).resolve().parent
W, H = 24.0, 16.9714285714
FONT = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf")
BOLD = fm.FontProperties(fname="C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams.update({
    "font.family": FONT.get_name(), "mathtext.fontset": "dejavusans",
    "svg.fonttype": "path", "axes.unicode_minus": False,
})
fig = plt.figure(figsize=(420/25.4, 297/25.4), dpi=363, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
C = dict(navy="#142B45", ink="#233D52", muted="#526E82",
         teal="#087F86", pale="#F1F8F8", line="#C8D6E1",
         blue="#3A638A", light="#F5F8FB", amber="#A65F25",
         warm="#FFF6EB", white="#FFFFFF")
texts, bounds, formulas = [], [], []
renderer = fig.canvas.get_renderer()

def tx(x, y, text, size=9.2, color=None, bold=False, width=None,
       height=None, align="left", math=False, min_size=7.9, tag=""):
    t = ax.text(x, y, text, fontsize=size, color=color or C["ink"],
                fontproperties=BOLD if bold else FONT, ha=align, va="top",
                linespacing=1.32, zorder=5)
    maxw = abs(ax.transData.transform((width or W, 0))[0] - ax.transData.transform((0, 0))[0])
    maxh = abs(ax.transData.transform((0, height or H))[1] - ax.transData.transform((0, 0))[1])
    while True:
        b = t.get_window_extent(renderer)
        if b.width <= maxw + .2 and b.height <= maxh + .2:
            break
        size -= .15
        if size < min_size:
            raise RuntimeError(f"Text cannot fit: {tag or text[:60]} w={b.width:.1f}/{maxw:.1f} h={b.height:.1f}/{maxh:.1f}")
        t.set_fontsize(size)
    texts.append(t)
    if width or height:
        bounds.append((t, x, y, width or W, height or H, align, tag))
    if math:
        formulas.append(text)
    return t

def box(x, y, w, h, fill=None, edge=None, lw=.8):
    p = FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0,rounding_size=0.10",
                      facecolor=fill or C["white"], edgecolor=edge or C["line"],
                      linewidth=lw, zorder=2)
    ax.add_patch(p)
    return p

def node(x,y,w,h,title,body,kind="normal",fs=8.9,titlefs=11.1):
    color=C["amber"] if kind=="weak" else C["teal"] if kind=="teal" else C["navy"]
    fill=C["warm"] if kind=="weak" else C["pale"] if kind=="teal" else C["white"]
    box(x,y,w,h,fill, C["amber"] if kind=="weak" else C["line"])
    tx(x+.12,y+.11,title,titlefs,color,True,width=w-.24,height=.32,tag=title+" title")
    if body:
        tx(x+.12,y+.47,body,fs,width=w-.24,height=h-.57,tag=title+" body")

def path(points, color=None, dashed=False, arrow=True, lw=1.0):
    code=[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)
    p=MPath(points,code)
    if arrow:
        patch=FancyArrowPatch(path=p, arrowstyle="-|>", mutation_scale=10,
                              linewidth=lw, edgecolor=color or C["muted"],
                              facecolor=color or C["muted"],
                              linestyle=(0,(3,3)) if dashed else "-", zorder=1)
    else:
        patch=PathPatch(p, fill=False, linewidth=lw, edgecolor=color or C["muted"],
                        linestyle=(0,(3,3)) if dashed else "-", capstyle="round",zorder=1)
    ax.add_patch(patch)

def rule(x,y,w):
    ax.plot([x,x+w],[y,y],color=C["line"],linewidth=.6,zorder=1)

def mathline(x,y,s,w=4.45,fs=10.7,h=.36):
    return tx(x,y,s,fs,width=w,height=h,math=True,min_size=8.0,tag=s)

# Header and columns.
ax.add_patch(Rectangle((0,0),W,1.36,color=C["navy"],zorder=0))
tx(.48,.23,"ERP 품목코드 생성 모델 — v6 아키텍처",24,C["white"],True,width=19.7,height=.57)
tx(.50,.87,"공유 특징  ·  6개 논리 head  ·  스키마 조건부 보정  ·  제약 조립  ·  Top-K 후보 평가",
   10.7,"#DEEBF1",width=19.4,height=.28)
tx(23.5,.31,"AS IMPLEMENTED\n2026-11-07",10.2,"#DEEBF1",True,width=3.0,height=.7,align="right")
box(.40,1.62,4.83,13.03,C["light"])
box(18.55,1.62,5.05,13.03,C["light"])
tx(.62,1.78,"A  학습 · 데이터 격리",13.8,C["navy"],True,width=4.4,height=.34)
tx(5.65,1.78,"B  추론 · 후보 생성 메커니즘",13.8,C["navy"],True,width=12.1,height=.34)
tx(18.78,1.78,"C  수식 · 평가 정의",13.8,C["navy"],True,width=4.5,height=.34)

# Training/data branch.
node(.62,2.25,2.57,1.02,"학습 입력","training.csv\n11,399행",fs=9.1,titlefs=10.5)
node(3.37,2.25,1.63,1.02,"평가 입력","inference.csv\n2,850행","weak",fs=8.6,titlefs=9.6)
node(.62,3.65,4.38,1.30,"중복·비코드 제외",
     "입력 중복 563행 · 비코드 2행 제외\n학습 참조 10,834행\n분류 충돌 205행 제외 → 10,629행")
path([(1.90,3.27),(1.90,3.65)])
path([(4.18,3.27),(4.18,3.47),(3.8,3.47),(3.8,3.65)],C["amber"],True)
tx(3.10,3.32,"입력 키 비교만",7.9,C["amber"],width=1.8,height=.21)
node(.62,5.28,4.38,1.24,"코드로부터 학습 라벨 생성",
     "분류: 실제 ERP 접두값 89종\n상세: 규칙 해석 가능한 1,931행\n동일 입력의 충돌 라벨은 head별 제외")
path([(2.81,4.95),(2.81,5.28)])
node(.62,6.86,4.38,1.63,"그룹 기반 3-fold 검증",
     "Group = norm(품목명, 규격명)\nTF-IDF도 각 fold의 학습 부분에서만 fit\n기본 head: OOF Macro F1 → 정확도 순 선택\n필드별: OOF 정확도로 XGB/LGB/평균 선택",fs=8.5)
path([(2.81,6.52),(2.81,6.86)])
mathline(.78,8.61,r"$\mathrm{MacroF1}=\frac{1}{|\mathcal{Y}|}\sum_{\ell\in\mathcal{Y}}F1_\ell$",4.05,10.5,.43)
node(.62,9.22,4.38,1.18,"전체 생성 정책 선택",
     "학습 내부 검증 639행 사용\n랭커·규칙·필드 보정·Router 조합 비교\n정책 선택 후 모델 가중치 고정",fs=8.7)
path([(2.81,9.04),(2.81,9.22)])
node(.62,10.79,4.38,1.17,"최종 모델 · 특징 저장",
     "XGB: 90 rounds / depth 3 / lr .08 · CUDA → CPU\nLGB: 90 / depth 4 / 12 leaves / lr .06 · CPU\nRF: 160 trees / balanced_subsample · CPU",fs=8.5)
path([(2.81,10.4),(2.81,10.79)])
box(.62,12.32,4.38,1.71,C["pale"])
tx(.76,12.43,"조회 · 평가 데이터 격리",11.0,C["teal"],True,width=4.10,height=.32)
tx(.76,12.87,"웹·CSV 참조: 학습 10,834행만 사용\n평가 입력 키 / 등록 코드 중복 모두 0건",8.7,width=4.1,height=.48)
mathline(.82,13.37,r"$\mathcal{K}_{tr}\cap\mathcal{K}_{ev}=\varnothing$",4.0,9.8,.27)
mathline(.82,13.71,r"$\mathcal{R}_{tr}^{code}\cap\mathcal{Y}_{ev}=\varnothing$",4.0,9.8,.27)
path([(2.81,11.96),(2.81,12.32)])
tx(.68,14.22,"업로드 CSV는 모델·조회 사전에 반영하지 않음",8.0,C["muted"],width=4.3,height=.24)

# Serving: shared preprocessing and six heads.
node(5.65,2.25,3.55,1.05,"단일 입력 / CSV 빈 코드",
     "품목명 + 규격명\n기존 코드는 보존 · 빈 코드만 예측",fs=8.8,titlefs=10.8)
node(9.50,2.25,3.55,1.05,"공통 전처리",
     "NFKC · 대문자 · 공백/하이픈 정리\n단위·치수·제조사·포장 증거 보존",fs=8.6,titlefs=10.8)
node(13.35,2.25,4.70,1.05,"공유 텍스트 · 구조화 특징 X",
     "char TF-IDF 2–4 gram · L2: 9,310 + 10,000\n+ 단위·치수·포장·제조사 구조화 특징",fs=8.9,titlefs=10.8)
path([(9.20,2.78),(9.50,2.78)])
path([(13.05,2.78),(13.35,2.78)])
tx(11.85,3.58,"동일 X와 원문 증거 재사용 · 필요한 head만 계산",8.6,C["muted"],width=11.9,height=.26,align="center")
heads=[
 ("분류","Random Forest\n160 trees\nn=10,629 / 89 labels","normal"),
 ("특성","XGBoost\nn=1,521\n201 labels","normal"),
 ("세분류 · 품번","XGBRank + LGBRank\nn=724 / groups=546\n702 unique targets\nmean rank 50:50","teal"),
 ("형상","XGBoost\nn=1,386\n119 labels","normal"),
 ("Packing","Random Forest\n160 trees\nn=1,423 / 48 labels","normal"),
 ("Vendor","XGBoost\nn=1,883\n21 labels","normal"),
]
centers=[]
for i,(title,body,kind) in enumerate(heads):
    x=5.65+i*2.09;centers.append(x+.955)
    node(x,4.38,1.91,1.65,title,body,kind,fs=8.3,titlefs=10.3)
path([(15.70,3.30),(15.70,4.03)],C["blue"])
path([(centers[0],4.03),(centers[-1],4.03)],C["blue"],arrow=False)
for x in centers:
    path([(x,4.03),(x,4.38)],C["blue"])
    path([(x,6.03),(x,6.56)],C["blue"],arrow=False)
path([(centers[0],6.56),(centers[-1],6.56)],C["blue"],arrow=False)
path([(11.85,6.56),(11.85,7.04)],C["blue"])
node(5.65,7.04,3.55,1.52,"코드 체계 Router",
     "품목군 규칙 + XGB (32체계)\n입력으로 적용 가능한 체계 제한\n점수 ≥0.4 시 선택; 그 외 규칙 경로",fs=8.4,titlefs=10.8)
node(9.50,7.04,4.70,1.52,"조건부 필드 보정 · 24 모델군",
     "shape / tol / volt / pack / vend …\n명시적 입력값·수치 변환을 우선 보존\n비명시 필드에 체계 조건부 모델 적용\nOOF 정확도 ≥0.6인 필드 모델군만 채택",fs=8.7,titlefs=10.8)
node(14.50,7.04,3.55,1.52,"회사 코드북 · 규칙 엔진",
     "36개 체계의 길이·허용값·위치\n입력 단위·별칭 → 코드로 변환\n고정 구간과 필수 필드 검사",fs=8.5,titlefs=10.6)
path([(9.50,3.30),(9.50,3.91),(5.42,3.91),(5.42,7.80),(5.65,7.80)],C["teal"],True)
path([(18.05,2.78),(18.30,2.78),(18.30,7.80),(18.05,7.80)],C["muted"],True)
path([(9.20,7.80),(9.50,7.80)],C["teal"],True)
path([(14.50,7.80),(14.20,7.80)],C["muted"],True)
path([(11.85,8.56),(11.85,8.91),(8.65,8.91),(8.65,9.28)],C["blue"])
path([(11.85,8.91),(15.05,8.91),(15.05,9.28)],C["amber"])
tx(5.78,8.99,"구간 규칙 확인",8.4,C["teal"],width=2.8,height=.22)
tx(12.18,8.99,"구간 규칙 미확인",8.4,C["amber"],width=2.8,height=.22)
node(5.65,9.28,6.00,1.75,"형식 검증 후보 · Constrained assembly",
     "6개 논리 값 → 체계별 실제 코드 위치로 배치\n구간 후보 조합 → Beam 80 → 길이·허용값 검사\n체계별 가변 길이 · 형식 검증과 적용 확정은 구분",fs=9.2,titlefs=11.1)
node(12.05,9.28,6.00,1.75,"미검증 추정 후보 · Exploratory expansion",
     "동일 모델·학습 참조 재사용 (별도 모델 학습 없음)\n학습 유사 코드 최대 40개 + 입력 식별자 치환\n관찰된 같은 길이의 코드 구간만 입력값으로 변환\n유사 근거가 없으면 학습 빈도 기반의 약한 후보\n6칸 참고값은 코드의 실제 구간으로 확정하지 않음",
     "weak",fs=8.7,titlefs=10.8)
path([(8.65,11.03),(8.65,11.33),(12.70,11.33),(12.70,11.67)],C["blue"])
path([(15.05,11.03),(15.05,11.67)],C["amber"])
node(5.65,11.67,4.00,1.17,"학습 전용 코드 검색",
     "정규화 입력 일치 / TF-IDF cosine\n품번 일치 + 품목군 보정\ninference 입력·등록 코드 중복 0건",fs=8.5,titlefs=10.5)
node(10.10,11.67,7.95,1.17,"후보 순위 · 중복 제거 · K ≤10",
     "주 결과와 미검증 추정 결과의 출처·검토 상태를 보존\n형식 검증 후보 / 미확인 경로의 확장 후보를 별도 표시",fs=9.2,titlefs=11.2)
path([(5.65,2.78),(5.30,2.78),(5.30,12.25),(5.65,12.25)],C["teal"],True)
path([(9.65,12.25),(10.10,12.25)],C["teal"])
path([(14.05,12.84),(14.05,13.20)],C["blue"])
node(5.65,13.20,12.40,1.08,"웹 / API / CSV 출력",
     "6개 칸 + 코드 후보 + 근거·검토 여부 | 전체 요청: context + 6 head + assembly = 8 events\n단일 칸은 해당 head만 | CSV: 빈 코드만 채움·기존 코드 보존·추정 후보와 참고값 별도 기록",
     fs=9.1,titlefs=10.9)
tx(5.75,14.43,"실선: 데이터·후보 흐름   ·   점선: 공유 증거·조건 참조   ·   주황: 적용 검증 전 추정 경로",
   8.0,C["muted"],width=12.10,height=.23)

# Equations: feature vector, conditioning, mixture, mean ranks, constraints, Hit@K.
mx,mw=18.80,4.55
def mh(y,s):
    tx(mx,y,s,10.1,C["teal"],True,width=mw,height=.25)
mh(2.20,"01  공유 특징")
mathline(mx,2.53,r"$\mathbf{x}=[\phi_n(n)\Vert\phi_s(s)\Vert\psi(n,s)]$",mw,11.0,.37)
mathline(mx,2.98,r"$\mathrm{idf}(g)=\log\frac{1+N_d}{1+\mathrm{df}(g)}+1$",mw,9.0,.32)
mh(3.39,"02  체계 선택 · 조건부 추정")
mathline(mx,3.71,r"$\hat{t}=\arg\max_{t\in\mathcal{T}(f)}p_\theta(t\mid\mathbf{x})$",mw,10.6,.38)
mathline(mx,4.15,r"$p_f(v\mid\mathbf{x},\hat{t}),\quad\mathbf{x}_f=[\mathbf{x}\Vert\mathrm{onehot}(\hat{t})]$",mw,10.0,.35)
tx(mx,4.58,"Router는 정답 체계를 입력받지 않음",8.1,C["muted"],width=mw,height=.24)
mh(4.98,"03  기본 분류기 결합")
mathline(mx,5.29,r"$p_j^{mix}(v\mid\mathbf{x})=\sum_m w_{jm}\,p_{jm}(v\mid\mathbf{x})$",mw,10.4,.38)
tx(mx,5.75,"w는 head별 OOF로 선택 · 현재 RF 또는 XGB 채택",8.1,C["muted"],width=mw,height=.27)
mh(6.14,"04  품번 후보 평균 순위")
mathline(mx,6.46,r"$r(c)=\frac{\mathrm{rank}(-s_X(c))+\mathrm{rank}(-s_L(c))}{2}$",mw,10.3,.43)
tx(mx,6.99,"XGB rank:pairwise / LGB lambdarank · 낮을수록 우선",8.1,C["muted"],width=mw,height=.24)
mathline(mx,7.23,r"$\mathbf{z}_c=[\mathbf{x}_{PN}\Vert\phi_c(c)\Vert\eta(c,n,s)]$",mw,9.9,.33)
mh(7.67,"05  구간 제약 · Beam 순위")
mathline(mx,7.99,r"$c=\mathrm{assemble}_{\hat{t}}(v_1,\ldots,v_6)$",mw,10.5,.32)
mathline(mx,8.39,r"$c\in\mathcal{C}_{\hat{t}}\ \Longleftrightarrow\ \mathrm{valid}_{\hat{t}}(c)=1$",mw,10.2,.34)
mathline(mx,8.80,r"$\widetilde{S}(c)=\log\widetilde{s}_1(v_1)+\cdots+\log\widetilde{s}_6(v_6)$",mw,10.5,.38)
tx(mx,9.24,"후보 점수는 순위용 · 코드 정답 확률은 미보정",8.1,C["muted"],width=mw,height=.25)
mh(9.67,"06  Top-K 정답 포함률")
mathline(mx,10.00,r"$\mathrm{Hit}@K=\frac{1}{N}\sum_{i=1}^{N}h_i(K)$",mw,10.6,.58)
mathline(mx,10.65,r"$h_i(K)=\mathbf{1}\{y_i\in\mathrm{unique}(\mathcal{C}_i)_{1:K}\}$",mw,10.0,.32)
tx(mx,11.04,"정답: CSV 원문 코드 · 그룹별 분모는 고정",8.1,C["muted"],width=mw,height=.24)
mh(11.40,"07  후보 수 통제")
mathline(mx,11.72,r"$\mathrm{AvgSize}@K=\frac{1}{N}\sum_i\min(K,|\mathrm{unique}(\mathcal{C}_i)|)$",mw,10.0,.43)
tx(mx,12.30,"진단 결과 | 원문 코드 일치",10.2,C["navy"],True,width=mw,height=.27)
table_y=12.68
ax.add_patch(Rectangle((mx,table_y),mw,.30,facecolor="#E6EEF3",edgecolor="none",zorder=2))
tx(mx+.06,table_y+.04,"평가 그룹 · 행 수",8.1,C["muted"],True,width=1.85,height=.21)
for xx,lab in [(21.12,"K=1"),(22.10,"K=5"),(23.08,"K=10")]:
    tx(xx,table_y+.04,lab,8.3,C["muted"],True,width=.8,height=.21,align="center")
for i,(lab,a,b,c) in enumerate([
    ("전체 · 2,850","9.68%","11.72%","12.25%"),
    ("규칙 해석 가능 · 473","58.35%","66.38%","68.50%"),
    ("미해석 · 2,377","0.00%","0.84%","1.05%"),
]):
    yy=table_y+.35+i*.34
    tx(mx+.06,yy,lab,8.0,width=1.90,height=.25)
    for xx,v in [(21.12,a),(22.10,b),(23.08,c)]:
        tx(xx,yy,v,8.5,C["ink"],width=.82,height=.25,align="center")
    rule(mx,yy+.29,mw)
tx(mx,14.05,"미해석 코드도 전체 문자열은 Top-K 채점 가능",8.1,C["muted"],width=mw,height=.25)

# Footer: concrete example + interpretation safeguards.
box(.40,15.00,23.20,1.55,C["light"])
tx(.63,15.16,"실제 위치 기반 조립 예시",11.0,C["navy"],True,width=11.7,height=.28)
parts=[("R","1",.66),("CP","2–3",1.10),("10","4–5",1.10),("F","6",.66),("1232","7–10",2.15),("TP","11–12",1.10),("Y","13",.66)]
xx=.64
for value,pos,ww in parts:
    box(xx,15.58,ww,.39,C["pale"],C["line"],.55)
    tx(xx+ww/2,15.61,value,12.1,C["navy"],True,width=ww-.08,height=.28,align="center")
    tx(xx+ww/2,16.01,pos,7.9,C["muted"],width=ww-.05,height=.20,align="center")
    xx+=ww+.08
tx(9.15,15.62,"RCP10F1232TPY",12.3,C["teal"],True,width=3.25,height=.30)
tx(.64,16.28,"분류 R  ·  특성 CPF  ·  품번 1232  ·  형상 10  ·  Packing TP  ·  Vendor Y",
   8.5,C["muted"],width=12.1,height=.22)
tx(13.13,15.16,"해석 · 운영 기준",11.0,C["navy"],True,width=10.1,height=.28)
tx(13.13,15.61,
   "입력 규칙·조회·모델의 결합 결과. 칸별 점수는 순수 모델 정확도와 구분.\n"
   "단일 등록 코드 완전 일치만 자동 확정. 미검증 추정·참고 head는 확정하지 않음.\n"
   "head 간 예측값 재입력은 없으며, 공유 코드 체계와 조립 제약이 출력들을 결합.\n"
   "inference는 학습·조회에서 제외; 규칙 대조에 이미 사용한 진단셋(완전 blind test 아님).",
   8.4,C["muted"],width=10.05,height=.86)
tx(.45,16.72,
   "SOURCE  preprocess.py · features.py · models.py · attribute_models.py · generation.py · spectrum.py · exploratory.py  |  snapshot: v6.0 / 2026-11-07",
   7.9,C["muted"],width=23.1,height=.19)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
audit=[]
for t,x,y,w,h,align,tag in bounds:
    bb=t.get_window_extent(renderer)
    if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
        raise RuntimeError("Canvas overflow: "+(tag or t.get_text()))
    audit.append({"tag":tag or t.get_text()[:80],"fontsize":t.get_fontsize(),"width_pixels":round(bb.width,1),"height_pixels":round(bb.height,1)})

stem=OUT/"erp-code-model-v6-onepage"
fig.savefig(stem.with_suffix(".png"),dpi=363,facecolor="white")
fig.savefig(stem.with_suffix(".svg"),facecolor="white")
# Explicit A3 print metadata without resampling the high-resolution raster.
with Image.open(stem.with_suffix(".png")) as im:
    im.save(stem.with_suffix(".png"),dpi=(363,363),optimize=True)
tree=ET.parse(stem.with_suffix(".svg"))
root=tree.getroot();root.set("width","420mm");root.set("height","297mm")
tree.write(stem.with_suffix(".svg"),encoding="utf-8",xml_declaration=True)

# Proof exports from the same figure, not a second visual design.
for label,x0,y0,x1,y1 in [
    ("training",.35,1.55,5.28,14.7),
    ("mechanism",5.25,1.55,18.45,14.7),
    ("equations",18.5,1.55,23.65,14.7),
]:
    sx,sy=fig.get_size_inches()
    region=Bbox.from_extents(x0/W*sx,(H-y1)/H*sy,x1/W*sx,(H-y0)/H*sy)
    fig.savefig(OUT/("qa-"+label+".png"),dpi=220,bbox_inches=region,facecolor="white",pad_inches=0)

report={"image":str(stem.with_suffix(".png")),"svg":str(stem.with_suffix(".svg")),
        "formula_count":len(formulas),"text_count":len(texts),
        "minimum_font_pt":min(t.get_fontsize() for t in texts),
        "text_bounds":audit,"formulas":formulas}
overlaps=[]
for i,a in enumerate(texts):
    ab=a.get_window_extent(renderer)
    for b in texts[i+1:]:
        bb=b.get_window_extent(renderer)
        wi=min(ab.x1,bb.x1)-max(ab.x0,bb.x0)
        he=min(ab.y1,bb.y1)-max(ab.y0,bb.y0)
        if wi>3 and he>3:
            overlaps.append([a.get_text()[:100],b.get_text()[:100],round(wi,1),round(he,1)])
report["text_overlaps"]=overlaps
with Image.open(stem.with_suffix(".png")) as im:report["pixel_size"]=list(im.size)
(OUT/"layout-audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in {"text_bounds","formulas"}},ensure_ascii=False,indent=2))
