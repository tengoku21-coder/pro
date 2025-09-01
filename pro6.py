# app.py
# -*- coding: utf-8 -*-
"""
충전기 수익성 분석 프로그램
- 시나리오 입력 전용
- 월별 사용량 증가율
- 계절별 TOU(경/중/최) 전력요금: 기본값 표 적용 (여름/봄·가을/겨울)
설치:
  pip install streamlit pandas numpy openpyxl
실행:
  streamlit run app.py
"""

from __future__ import annotations
import math, io
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st

# ----------------- 유틸 -----------------
def fmt_won(x: float) -> str:
    try: return f"{int(round(x)):,} 원"
    except: return "-"
def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))): return "-"
    return f"{x*100:.2f}%"
def ann_from_monthly(r_m: float) -> float: return (1+r_m)**12-1
def monthly_from_annual(r_a: float) -> float: return (1+r_a)**(1/12)-1
def payback_period_months(cashflows: List[float]) -> Optional[int]:
    cum=0.0
    for i,cf in enumerate(cashflows):
        cum+=cf
        if cum>=0: return i
    return None
def npv(cashflows: List[float],r_m:float)->float:
    return sum(cf/((1+r_m)**t) for t,cf in enumerate(cashflows))
def irr_bisection(cashflows: List[float],lo=-0.9,hi=5.0,tol=1e-7,max_iter=200)->Optional[float]:
    def f(r): return npv(cashflows,r)
    f_lo,f_hi=f(lo),f(hi)
    if math.isnan(f_lo) or math.isnan(f_hi) or f_lo*f_hi>0: return None
    for _ in range(max_iter):
        mid=(lo+hi)/2; f_mid=f(mid)
        if abs(f_mid)<tol: return mid
        if f_lo*f_mid<0: hi,f_hi=mid,f_mid
        else: lo,f_lo=mid,f_mid
    return (lo+hi)/2

# ----------------- 보조금 -----------------
def per_unit_subsidy_by_count(total_units:int)->int:
    if total_units<=0: return 0
    if total_units==1: return 2_200_000
    if 2<=total_units<=5: return 2_000_000
    return 2_200_000

# ----------------- 데이터 구조 -----------------
@dataclass
class Inputs:
    total_units:int
    charger_capacity_kw:float
    monthly_kwh_per_unit_start:float
    car_to_charger_ratio:float
    monthly_growth_rate:float
    growth_months:int
    kwh_cap:float

    sell_price_normal:float
    sell_price_promo:float
    promo_months:int

    # 계절별 TOU 요금(원/kWh)
    off_summer:float; mid_summer:float; peak_summer:float
    off_springfall:float; mid_springfall:float; peak_springfall:float
    off_winter:float; mid_winter:float; peak_winter:float
    # 연중 동일 비중(0~1)
    tou_share_off:float; tou_share_mid:float; tou_share_peak:float

    base_power_fee_per_kw:float
    comm_fee:float; opex_maint:float; network_fee:float

    unit_purchase_cost:float
    kepco_contribution_per_count:float
    kepco_contribution_count:int
    meter_split_per_count:float

    subsidy_manual_override:bool
    subsidy_manual_per_unit:float

    analysis_years:int
    discount_rate_annual:float

# ----------------- 사용량 시퀀스 -----------------
def build_kwh_series(start_kwh: float, months: int, monthly_growth_rate: float,
                     growth_months: int, kwh_cap: float, ratio: float) -> np.ndarray:
    g = monthly_growth_rate
    m = months
    seq = np.empty(m, dtype=float)
    apply_months = m if (growth_months is None or growth_months<=0 or growth_months>m) else growth_months
    for t in range(apply_months):
        val = start_kwh * ((1+g) ** t)
        if kwh_cap and kwh_cap>0: val = min(val, kwh_cap)
        seq[t] = val
    if apply_months < m:
        hold_val = seq[apply_months-1] if apply_months>0 else start_kwh
        seq[apply_months:] = hold_val
    seq *= (ratio if ratio>0 else 1.0)
    return seq

def normalize_shares(off: float, mid: float, peak: float) -> Tuple[float,float,float]:
    s = off + mid + peak
    if s <= 0: return 1.0, 0.0, 0.0
    return off/s, mid/s, peak/s

def season_of_month(m: int) -> str:
    """m: 1~12 → 'winter'|'springfall'|'summer'"""
    if m in (6,7,8): return "summer"
    if m in (3,4,5,9,10): return "springfall"
    return "winter"  # 11,12,1,2

# ----------------- 계산 -----------------
def build_cashflows(inp:Inputs)->Tuple[pd.DataFrame,List[float],float,Optional[float],Optional[int],Optional[float],float]:
    months=inp.analysis_years*12
    mrate=monthly_from_annual(inp.discount_rate_annual)

    # CAPEX
    capex_chargers=inp.unit_purchase_cost*inp.total_units
    kepco_cnt=max(0,min(inp.kepco_contribution_count,inp.total_units))
    split_cnt=max(0,inp.total_units-kepco_cnt)
    capex_total=capex_chargers+inp.kepco_contribution_per_count*kepco_cnt+inp.meter_split_per_count*split_cnt
    per_unit_sub=inp.subsidy_manual_per_unit if inp.subsidy_manual_override else per_unit_subsidy_by_count(inp.total_units)
    subsidy_total=per_unit_sub*inp.total_units
    initial_outflow=capex_total-subsidy_total

    # 월별 kWh/대
    kwh_series = build_kwh_series(
        start_kwh=inp.monthly_kwh_per_unit_start,
        months=months,
        monthly_growth_rate=inp.monthly_growth_rate,
        growth_months=inp.growth_months,
        kwh_cap=inp.kwh_cap,
        ratio=inp.car_to_charger_ratio
    )

    # 비중 정규화(연중 동일)
    sh_off, sh_mid, sh_peak = normalize_shares(inp.tou_share_off, inp.tou_share_mid, inp.tou_share_peak)

    base_fee_per_unit=inp.charger_capacity_kw*inp.base_power_fee_per_kw
    other_opex_per_unit=inp.comm_fee+inp.opex_maint+inp.network_fee

    rows=[]; cashflows=[-initial_outflow]
    for t in range(1, months+1):
        kwh_unit = kwh_series[t-1]

        # 판매단가(프로모션 반영)
        sell_price = inp.sell_price_promo if (inp.promo_months>0 and t<=inp.promo_months and inp.sell_price_promo>0) else inp.sell_price_normal

        # 계절별 요금 선택
        season = season_of_month(((t-1)%12)+1)
        if season=="summer":
            r_off, r_mid, r_peak = inp.off_summer, inp.mid_summer, inp.peak_summer
        elif season=="springfall":
            r_off, r_mid, r_peak = inp.off_springfall, inp.mid_springfall, inp.peak_springfall
        else:  # winter
            r_off, r_mid, r_peak = inp.off_winter, inp.mid_winter, inp.peak_winter

        # TOU 분해 (대당)
        kwh_off  = kwh_unit*sh_off
        kwh_mid  = kwh_unit*sh_mid
        kwh_peak = kwh_unit*sh_peak

        energy_cost_unit_off  = kwh_off*r_off
        energy_cost_unit_mid  = kwh_mid*r_mid
        energy_cost_unit_peak = kwh_peak*r_peak
        energy_cost_unit = energy_cost_unit_off + energy_cost_unit_mid + energy_cost_unit_peak

        revenue_unit = kwh_unit * sell_price

        # 전체(사이트)
        revenue_total = revenue_unit * inp.total_units
        energy_off_total  = energy_cost_unit_off  * inp.total_units
        energy_mid_total  = energy_cost_unit_mid  * inp.total_units
        energy_peak_total = energy_cost_unit_peak * inp.total_units
        energy_total = energy_off_total + energy_mid_total + energy_peak_total

        base_fee_total = base_fee_per_unit * inp.total_units
        other_opex_total = other_opex_per_unit * inp.total_units

        total_opex_month = energy_total + base_fee_total + other_opex_total
        net_cash = revenue_total - total_opex_month

        cashflows.append(net_cash)
        rows.append({
            "월": t,
            "계절": {"summer":"여름","springfall":"봄·가을","winter":"겨울"}[season],
            "사용전력(kWh/대)": kwh_unit,
            "판매단가(원/kWh)": sell_price,
            "경부하비용(합계)": energy_off_total,
            "중간부하비용(합계)": energy_mid_total,
            "최대부하비용(합계)": energy_peak_total,
            "전력량요금(합계)": energy_total,
            "기본요금(합계)": base_fee_total,
            "기타OPEX(합계)": other_opex_total,
            "매출(합계)": revenue_total,
            "총비용(합계)": total_opex_month,
            "순현금흐름": net_cash
        })

    df=pd.DataFrame(rows)
    payback_m=payback_period_months(cashflows)
    irr_m=irr_bisection(cashflows)
    irr_a=ann_from_monthly(irr_m) if irr_m is not None else None
    npv_val=npv(cashflows,mrate)
    return df,cashflows,npv_val,irr_m,payback_m,irr_a,initial_outflow

# ----------------- UI -----------------
st.set_page_config(page_title="충전기 수익성 분석",page_icon="⚡",layout="wide")
st.title("⚡ 충전기 수익성 분석 프로그램")
st.caption("· ‘한전불입 대수 + 모자분리 대수 = 총 대수’ 자동 · 월별 사용량 증가율 · 계절별 TOU 요금 기본값 적용")

scenario_count=st.number_input("비교할 시나리오 수",1,5,3,1)
tabs=st.tabs([f"시나리오 {i+1}" for i in range(scenario_count)])
scenario_results:Dict[str,Dict]={}

for i,tab in enumerate(tabs,start=1):
    with tab:
        st.markdown(f"### 시나리오 {i}")

        # 규모/사용량
        total_units=st.number_input(f"[S{i}] 총 대수",1,1000,6,1,key=f"units{i}")
        kepco_cnt=st.number_input(f"[S{i}] 한전불입 대수",0,total_units,0,1,key=f"kepco{i}")
        split_cnt=total_units-kepco_cnt
        st.info(f"모자분리 대수 = {split_cnt} (자동)")

        charger_capacity_kw=st.number_input(f"[S{i}] 충전기 용량(kW/대)",1.0,500.0,7.0,0.5,key=f"cap{i}")

        colA, colB = st.columns(2)
        with colA:
            monthly_kwh_start=st.number_input(f"[S{i}] 초기 월평균 사용량(kWh/대)",0.0,10000.0,180.0,10.0,key=f"kwh0_{i}")
            monthly_growth_rate=st.number_input(f"[S{i}] 월별 증가율(%)",0.0,100.0,0.0,0.5,key=f"grow{i}")/100.0
        with colB:
            growth_months=st.number_input(f"[S{i}] 증가율 적용 개월수(0=전체)",0,360,0,1,key=f"gmons{i}")
            kwh_cap=st.number_input(f"[S{i}] 최대치 kWh/대(0=무제한)",0.0,10000.0,0.0,10.0,key=f"kcap{i}")

        ratio=st.number_input(f"[S{i}] 차충비(가중치)",0.1,10.0,1.0,0.1,key=f"ratio{i}")

        # 판매요금(단일/프로모션)
        sell=st.number_input(f"[S{i}] 충전요금(원/kWh)",0.0,5000.0,350.0,10.0,key=f"sell{i}")
        promo=st.number_input(f"[S{i}] 프로모션요금(원/kWh)",0.0,5000.0,250.0,10.0,key=f"promo{i}")
        promo_m=st.number_input(f"[S{i}] 프로모션개월",0,120,0,1,key=f"promo_m{i}")

        # 계절별 TOU 요금 기본값 + 수정 가능
        with st.expander(f"[S{i}] 계절별 전력요금(원/kWh) — 기본값 표 적용"):
            st.caption("여름(6–8월) / 봄·가을(3–5, 9–10월) / 겨울(11–2월)")
            col1,col2,col3 = st.columns(3)
            with col1:
                off_summer = st.number_input(f"여름·경부하 [S{i}]", 0.0, 5000.0, 95.9, 0.1, key=f"off_su{i}")
                mid_summer = st.number_input(f"여름·중간부하 [S{i}]", 0.0, 5000.0, 162.2, 0.1, key=f"mid_su{i}")
                peak_summer= st.number_input(f"여름·최대부하 [S{i}]", 0.0, 5000.0, 203.5, 0.1, key=f"peak_su{i}")
            with col2:
                off_sf = st.number_input(f"봄·가을·경부하 [S{i}]", 0.0, 5000.0, 85.4, 0.1, key=f"off_sf{i}")
                mid_sf = st.number_input(f"봄·가을·중간부하 [S{i}]", 0.0, 5000.0, 97.2, 0.1, key=f"mid_sf{i}")
                peak_sf= st.number_input(f"봄·가을·최대부하 [S{i}]", 0.0, 5000.0, 102.1, 0.1, key=f"peak_sf{i}")
            with col3:
                off_wi = st.number_input(f"겨울·경부하 [S{i}]", 0.0, 5000.0, 110.6, 0.1, key=f"off_wi{i}")
                mid_wi = st.number_input(f"겨울·중간부하 [S{i}]", 0.0, 5000.0, 143.1, 0.1, key=f"mid_wi{i}")
                peak_wi= st.number_input(f"겨울·최대부하 [S{i}]", 0.0, 5000.0, 172.0, 0.1, key=f"peak_wi{i}")

        # 연중 동일 비중(조정 가능)
        with st.expander(f"[S{i}] TOU 사용 비중(%) — 연중 동일, 합계 자동정규화"):
            colx,coly,colz = st.columns(3)
            with colx:
                tou_share_off=st.number_input(f"경부하 비중(%) [S{i}]", 0.0, 100.0, 40.0, 1.0, key=f"s_off{i}")/100.0
            with coly:
                tou_share_mid=st.number_input(f"중간부하 비중(%) [S{i}]", 0.0, 100.0, 40.0, 1.0, key=f"s_mid{i}")/100.0
            with colz:
                tou_share_peak=st.number_input(f"최대부하 비중(%) [S{i}]", 0.0, 100.0, 20.0, 1.0, key=f"s_peak{i}")/100.0
            ssum = tou_share_off + tou_share_mid + tou_share_peak
            if abs(ssum-1.0)>1e-6:
                st.caption(f"※ 현재 합계 {ssum*100:.1f}% → 계산 시 자동 정규화")

        # 고정비
        basefee=st.number_input(f"[S{i}] 기본전력요금(원/kW·월)",0.0,10000.0,2390.0,10.0,key=f"base{i}")
        comm=st.number_input(f"[S{i}] 통신비(원/대/월)",0.0,500000.0,5000.0,500.0,key=f"comm{i}")
        maint=st.number_input(f"[S{i}] 유지보수비(원/대/월)",0.0,500000.0,5000.0,500.0,key=f"mnt{i}")
        netf=st.number_input(f"[S{i}] 네트워크비(원/대/월)",0.0,500000.0,5000.0,500.0,key=f"net{i}")

        # CAPEX
        unit_cost=st.number_input(f"[S{i}] 충전기 구매비(원/대)",0.0,100000000.0,1800000.0,50000.0,key=f"ucost{i}")
        kepco_cost=st.number_input(f"[S{i}] 한전불입 비용(원/대)",0.0,100000000.0,0.0,50000.0,key=f"kcost{i}")
        split_cost=st.number_input(f"[S{i}] 모자분리 비용(원/대)",0.0,100000000.0,0.0,50000.0,key=f"scost{i}")

        # 보조금/재무
        sub_over=st.checkbox(f"[S{i}] 보조금 수동입력",key=f"subo{i}")
        sub_val=st.number_input(f"[S{i}] 보조금(원/대, 수동)",0.0,10000000.0,2000000.0,50000.0,key=f"subv{i}",disabled=not sub_over)
        yrs=st.number_input(f"[S{i}] 분석기간(년)",1,50,10,1,key=f"yrs{i}")
        disc=st.number_input(f"[S{i}] 할인율(연,% )",0.0,50.0,8.0,0.5,key=f"disc{i}")/100

        # 입력 객체
        inp=Inputs(
            total_units=int(total_units),
            charger_capacity_kw=charger_capacity_kw,
            monthly_kwh_per_unit_start=monthly_kwh_start,
            car_to_charger_ratio=ratio,
            monthly_growth_rate=monthly_growth_rate,
            growth_months=int(growth_months),
            kwh_cap=kwh_cap,

            sell_price_normal=sell,
            sell_price_promo=promo,
            promo_months=int(promo_m),

            off_summer=off_summer, mid_summer=mid_summer, peak_summer=peak_summer,
            off_springfall=off_sf, mid_springfall=mid_sf, peak_springfall=peak_sf,
            off_winter=off_wi, mid_winter=mid_wi, peak_winter=peak_wi,

            tou_share_off=tou_share_off, tou_share_mid=tou_share_mid, tou_share_peak=tou_share_peak,

            base_power_fee_per_kw=basefee,
            comm_fee=comm, opex_maint=maint, network_fee=netf,

            unit_purchase_cost=unit_cost,
            kepco_contribution_per_count=kepco_cost,
            kepco_contribution_count=int(kepco_cnt),
            meter_split_per_count=split_cost,

            subsidy_manual_override=sub_over,
            subsidy_manual_per_unit=sub_val,

            analysis_years=int(yrs),
            discount_rate_annual=disc
        )

        # 계산
        df,cfs,npv_val,irr_m,pb_m,irr_a,init=build_cashflows(inp)

        # KPI
        c1,c2,c3,c4=st.columns(4)
        c1.metric("초기투자(보조금 반영)",fmt_won(init))
        c2.metric("NPV",fmt_won(npv_val))
        c3.metric("IRR(월)",fmt_pct(irr_m))
        c4.metric("IRR(연환산)",fmt_pct(irr_a))
        st.metric("회수기간", f"{pb_m}개월 ({pb_m/12:.1f}년)" if pb_m else "미회수")

        # 표
        st.markdown("#### 월별 현금흐름 (합계 & TOU 분해)")
        df_show = df.copy()
        money_cols = ["경부하비용(합계)","중간부하비용(합계)","최대부하비용(합계)",
                      "전력량요금(합계)","기본요금(합계)","기타OPEX(합계)","매출(합계)","총비용(합계)","순현금흐름"]
        for c in money_cols: df_show[c]=df_show[c].map(fmt_won)
        st.dataframe(df_show,use_container_width=True,height=380)

        # 사용량 프리뷰
        st.markdown("##### 사용량 변화(앞 24개월, kWh/대)")
        st.dataframe(df[["월","계절","사용전력(kWh/대)"]].head(24),use_container_width=True,height=240)

        scenario_results[f"시나리오{i}"]={"Inputs":inp,"DF":df,"NPV":npv_val,
                                          "IRR_M":irr_m,"IRR_A":irr_a,"PB":pb_m,"Init":init}

# ----------------- 시나리오 비교 -----------------
st.divider()
st.subheader("시나리오 비교 요약")
rows=[]
for k,v in scenario_results.items():
    inp=v["Inputs"]
    rows.append({
        "시나리오":k, "총대수":inp.total_units, "초기투자":v["Init"], "NPV":v["NPV"],
        "IRR(월)":v["IRR_M"], "IRR(연)":v["IRR_A"], "회수개월":v["PB"],
        "초기kWh/대":inp.monthly_kwh_per_unit_start, "월증가율(%)":inp.monthly_growth_rate*100,
        "증가개월":inp.growth_months, "kWh상한":inp.kwh_cap,
        "경/중/최 비중(%)": f"{inp.tou_share_off*100:.0f}/{inp.tou_share_mid*100:.0f}/{inp.tou_share_peak*100:.0f}",
        "기본요금(원/kW)": int(inp.base_power_fee_per_kw)
    })
df_sum=pd.DataFrame(rows)
df_sum_show=df_sum.copy()
for c in ["초기투자","NPV"]: df_sum_show[c]=df_sum_show[c].map(fmt_won)
for c in ["IRR(월)","IRR(연)"]: df_sum_show[c]=df_sum_show[c].map(fmt_pct)
df_sum_show["회수개월"]=df_sum_show["회수개월"].apply(lambda x:"-" if pd.isna(x) else f"{int(x)}")
st.dataframe(df_sum_show,use_container_width=True)

# ----------------- 엑셀 다운로드 -----------------
def inputs_to_df(inp:Inputs)->pd.DataFrame:
    d=asdict(inp); return pd.DataFrame({"항목":list(d.keys()),"값":list(d.values())})
buffer=io.BytesIO()
with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
    df_sum.to_excel(writer,index=False,sheet_name="Summary")
    for k,v in scenario_results.items():
        inputs_to_df(v["Inputs"]).to_excel(writer,index=False,sheet_name=f"{k}-Inputs")
        v["DF"].to_excel(writer,index=False,sheet_name=f"{k}-Monthly")
st.download_button("엑셀 다운로드 (요약+시나리오별 상세)",
                   data=buffer.getvalue(),
                   file_name="charger_scenarios_TOU_seasonal.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("※ IRR은 월 현금흐름 기준, 연환산은 (1+월IRR)^12 - 1.")
st.caption("※ 기본전력요금은 ‘충전기 용량 × 기본전력요금(원/kW·월)’ 단순 모델이며 실제 계약전력/피크요금 구조와 차이가 있을 수 있습니다.")
