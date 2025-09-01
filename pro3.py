# app.py
# -*- coding: utf-8 -*-
"""
충전기 수익성 분석 프로그램 (사이드바 제거 / 시나리오 입력 전용)

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

# ---------- 유틸 ----------
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

# ---------- 보조금 ----------
def per_unit_subsidy_by_count(total_units:int)->int:
    if total_units<=0: return 0
    if total_units==1: return 2_200_000
    if 2<=total_units<=5: return 2_000_000
    return 2_200_000

# ---------- 데이터 구조 ----------
@dataclass
class Inputs:
    total_units:int; charger_capacity_kw:float; monthly_kwh_per_unit:float; car_to_charger_ratio:float
    sell_price_normal:float; sell_price_promo:float; promo_months:int; grid_energy_rate:float; base_power_fee_per_kw:float
    comm_fee:float; opex_maint:float; network_fee:float
    unit_purchase_cost:float; kepco_contribution_per_count:float; kepco_contribution_count:int; meter_split_per_count:float
    subsidy_manual_override:bool; subsidy_manual_per_unit:float
    analysis_years:int; discount_rate_annual:float

# ---------- 계산 ----------
def build_cashflows(inp:Inputs)->Tuple[pd.DataFrame,List[float],float,Optional[float],Optional[int],Optional[float],float]:
    months=inp.analysis_years*12; mrate=monthly_from_annual(inp.discount_rate_annual)
    capex_chargers=inp.unit_purchase_cost*inp.total_units
    kepco_cnt=max(0,min(inp.kepco_contribution_count,inp.total_units))
    split_cnt=max(0,inp.total_units-kepco_cnt)
    capex_total=capex_chargers+inp.kepco_contribution_per_count*kepco_cnt+inp.meter_split_per_count*split_cnt
    per_unit_sub=inp.subsidy_manual_per_unit if inp.subsidy_manual_override else per_unit_subsidy_by_count(inp.total_units)
    subsidy_total=per_unit_sub*inp.total_units
    initial_outflow=capex_total-subsidy_total
    kwh_per_unit=inp.monthly_kwh_per_unit*(inp.car_to_charger_ratio if inp.car_to_charger_ratio>0 else 1.0)
    base_fee_per_unit=inp.charger_capacity_kw*inp.base_power_fee_per_kw
    other_opex_per_unit=inp.comm_fee+inp.opex_maint+inp.network_fee
    rows=[]; cashflows=[-initial_outflow]
    for m in range(1,months+1):
        sell_price=inp.sell_price_promo if (inp.promo_months>0 and m<=inp.promo_months and inp.sell_price_promo>0) else inp.sell_price_normal
        revenue_per_unit=kwh_per_unit*sell_price
        energy_cost_per_unit=kwh_per_unit*inp.grid_energy_rate
        total_cost_per_unit=base_fee_per_unit+energy_cost_per_unit+other_opex_per_unit
        revenue_total=revenue_per_unit*inp.total_units
        total_opex_month=(energy_cost_per_unit+base_fee_per_unit+other_opex_per_unit)*inp.total_units
        net_cash=revenue_total-total_opex_month
        cashflows.append(net_cash)
        rows.append({"월":m,"판매단가":sell_price,"사용전력(kWh/대)":kwh_per_unit,
                     "매출(합계)":revenue_total,"총비용(합계)":total_opex_month,"순현금흐름":net_cash})
    df=pd.DataFrame(rows)
    payback_m=payback_period_months(cashflows); irr_m=irr_bisection(cashflows)
    irr_a=ann_from_monthly(irr_m) if irr_m is not None else None
    npv_val=npv(cashflows,mrate)
    return df,cashflows,npv_val,irr_m,payback_m,irr_a,initial_outflow

# ---------- UI ----------
st.set_page_config(page_title="충전기 수익성 분석",page_icon="⚡",layout="wide")
st.title("⚡ 충전기 수익성 분석 프로그램")
scenario_count=st.number_input("비교할 시나리오 수",1,5,3,1)
tabs=st.tabs([f"시나리오 {i+1}" for i in range(scenario_count)])
scenario_results:Dict[str,Dict]={}

for i,tab in enumerate(tabs,start=1):
    with tab:
        st.markdown(f"### 시나리오 {i}")
        total_units=st.number_input(f"[S{i}] 총 대수",1,100,6,1,key=f"units{i}")
        kepco_cnt=st.number_input(f"[S{i}] 한전불입 대수",0,total_units,0,1,key=f"kepco{i}")
        split_cnt=total_units-kepco_cnt
        st.info(f"모자분리 대수 = {split_cnt} (자동)")
        charger_capacity_kw=st.number_input(f"[S{i}] 충전기 용량(kW)",1.0,200.0,7.0,0.5,key=f"cap{i}")
        monthly_kwh=st.number_input(f"[S{i}] 월평균 충전량(kWh/대)",0.0,5000.0,180.0,10.0,key=f"kwh{i}")
        ratio=st.number_input(f"[S{i}] 차충비",0.1,5.0,1.0,0.1,key=f"ratio{i}")
        sell=st.number_input(f"[S{i}] 충전요금(원/kWh)",0.0,2000.0,350.0,10.0,key=f"sell{i}")
        promo=st.number_input(f"[S{i}] 프로모션요금",0.0,2000.0,250.0,10.0,key=f"promo{i}")
        promo_m=st.number_input(f"[S{i}] 프로모션개월",0,60,0,1,key=f"promo_m{i}")
        grid=st.number_input(f"[S{i}] 전력량요금",0.0,1000.0,120.0,5.0,key=f"grid{i}")
        basefee=st.number_input(f"[S{i}] 기본전력요금(원/kW·월)",0.0,5000.0,2390.0,10.0,key=f"base{i}")
        comm=st.number_input(f"[S{i}] 통신비(원/대/월)",0.0,50000.0,5000.0,500.0,key=f"comm{i}")
        maint=st.number_input(f"[S{i}] 유지보수비",0.0,50000.0,5000.0,500.0,key=f"mnt{i}")
        netf=st.number_input(f"[S{i}] 네트워크비",0.0,50000.0,5000.0,500.0,key=f"net{i}")
        unit_cost=st.number_input(f"[S{i}] 충전기 구매비(원/대)",0.0,10000000.0,1800000.0,50000.0,key=f"ucost{i}")
        kepco_cost=st.number_input(f"[S{i}] 한전불입 비용(원/대)",0.0,10000000.0,0.0,50000.0,key=f"kcost{i}")
        split_cost=st.number_input(f"[S{i}] 모자분리 비용(원/대)",0.0,10000000.0,0.0,50000.0,key=f"scost{i}")
        sub_over=st.checkbox(f"[S{i}] 보조금 수동입력",key=f"subo{i}")
        sub_val=st.number_input(f"[S{i}] 보조금(원/대)",0.0,3000000.0,2000000.0,50000.0,key=f"subv{i}",disabled=not sub_over)
        yrs=st.number_input(f"[S{i}] 분석기간(년)",1,30,10,1,key=f"yrs{i}")
        disc=st.number_input(f"[S{i}] 할인율(연,% )",0.0,30.0,8.0,0.5,key=f"disc{i}")/100
        inp=Inputs(total_units,charger_capacity_kw,monthly_kwh,ratio,sell,promo,int(promo_m),grid,basefee,
                   comm,maint,netf,unit_cost,kepco_cost,int(kepco_cnt),split_cost,
                   sub_over,sub_val,int(yrs),disc)
        df,cfs,npv_val,irr_m,pb_m,irr_a,init=build_cashflows(inp)
        st.metric("초기투자",fmt_won(init)); st.metric("NPV",fmt_won(npv_val))
        st.metric("IRR(월)",fmt_pct(irr_m)); st.metric("IRR(연)",fmt_pct(irr_a))
        if pb_m: st.metric("회수기간",f"{pb_m}개월 ({pb_m/12:.1f}년)")
        else: st.metric("회수기간","미회수")
        st.dataframe(df,use_container_width=True,height=320)
        scenario_results[f"시나리오{i}"]={"Inputs":inp,"DF":df,"NPV":npv_val,
                                          "IRR_M":irr_m,"IRR_A":irr_a,"PB":pb_m,"Init":init}

# ---------- 시나리오 비교 ----------
st.subheader("시나리오 비교 요약")
rows=[]
for k,v in scenario_results.items():
    rows.append({"시나리오":k,"총대수":v["Inputs"].total_units,"초기투자":v["Init"],
                 "NPV":v["NPV"],"IRR(월)":v["IRR_M"],"IRR(연)":v["IRR_A"],"회수개월":v["PB"]})
df_sum=pd.DataFrame(rows)
df_sum_show=df_sum.copy()
df_sum_show["초기투자"]=df_sum["초기투자"].map(fmt_won)
df_sum_show["NPV"]=df_sum["NPV"].map(fmt_won)
df_sum_show["IRR(월)"]=df_sum["IRR(월)"].map(fmt_pct)
df_sum_show["IRR(연)"]=df_sum["IRR(연)"].map(fmt_pct)
df_sum_show["회수개월"]=df_sum["회수개월"].apply(lambda x:"-" if pd.isna(x) else f"{int(x)}")
st.dataframe(df_sum_show,use_container_width=True)

# ---------- 엑셀 다운로드 ----------
def inputs_to_df(inp:Inputs)->pd.DataFrame:
    d=asdict(inp); return pd.DataFrame({"항목":list(d.keys()),"값":list(d.values())})
buffer=io.BytesIO()
with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
    df_sum.to_excel(writer,index=False,sheet_name="Summary")
    for k,v in scenario_results.items():
        inputs_to_df(v["Inputs"]).to_excel(writer,index=False,sheet_name=f"{k}-Inputs")
        v["DF"].to_excel(writer,index=False,sheet_name=f"{k}-Monthly")
st.download_button("엑셀 다운로드",data=buffer.getvalue(),
                   file_name="charger_scenarios.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
