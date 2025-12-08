import streamlit as st
import pandas as pd
import numpy_financial as npf
import altair as alt

# 페이지 기본 설정
st.set_page_config(page_title="태성콘텍 충전인프라 월별 수익성 분석", layout="wide")

st.title("⚡ 태성콘텍 충전사업 시뮬레이션 (월별 상세 분석)")
st.markdown("---")

# ==========================================
# [사이드바] 변수 입력 컨트롤 패널
# ==========================================
st.sidebar.header("🛠️ 시뮬레이션 설정")

# 0. 시뮬레이션 기간 및 상환 설정 (수정된 부분)
# 주의: with 구문 안에서는 st.sidebar.slider가 아닌 st.slider를 써야 탭 안에 들어갑니다.
with st.sidebar.expander("0. 기간 및 상환(Exit) 설정", expanded=True):
    st.write("⏳ **기간 설정**")
    simulation_years = st.slider("총 시뮬레이션 기간(년)", min_value=1, max_value=20, value=7)
    # 월 단위 변환
    total_months = simulation_years * 12
    
    st.markdown("---")
    st.write("💰 **원금 상환(Exit) 시나리오**")
    use_repayment = st.checkbox("투자원금 상환 포함", value=True)
    
    if use_repayment:
        # 상환 연도는 총 기간을 넘을 수 없도록 동적 설정
        repayment_year = st.slider("원금 상환 시점(년차)", 
                                   min_value=1, 
                                   max_value=simulation_years, 
                                   value=min(5, simulation_years),
                                   help="해당 년차의 마지막 달(12월)에 상환합니다.")
        repayment_month_idx = repayment_year * 12
    else:
        repayment_year = None
        repayment_month_idx = None

# 1. 자금조달 및 비용
with st.sidebar.expander("1. 자금조달 및 투자비용", expanded=False):
    infra_cost = st.number_input("충전인프라 투자비(원/1기)", value=2700000, step=100000)
    charger_cost = st.number_input("충전기 비용(원/1기)", value=600000, step=50000)
    subsidy = st.number_input("보조금(원/1기)", value=1800000, step=100000)
    num_units = st.number_input("설치 대수(기)", value=1, step=1)
    
    investment_amount = st.number_input("투자유치 금액(채권)", value=2000000, step=100000)

# 2. 단계별 기간 설정 (Phase)
with st.sidebar.expander("2. 이익 배분 단계 설정", expanded=False):
    # 1단계
    p1_years = st.slider("1단계 기간(년) - 이자 지급", 1, 5, 3)
    p1_rate_annual = st.slider("1단계 연이자율(%)", 0.0, 20.0, 5.0) / 100.0
    
    # 2단계
    p2_years = st.slider("2단계 기간(년) - 이익 배분", 1, 5, 2)
    p2_share = st.slider("2단계 투자자 배분율(%)", 0, 100, 50) / 100.0
    
    # 3단계 안내
    p3_start_year = p1_years + p2_years + 1
    st.caption(f"💡 3단계(회사 독점)는 {p3_start_year}년차부터 적용됩니다.")

# 3. 매출 및 운영 설정
with st.sidebar.expander("3. 매출 및 운영 변수", expanded=False):
    promo_months = st.slider("프로모션 기간(개월)", 0, 12, 6)
    promo_price = st.number_input("프로모션 요금(원/kWh)", value=168)
    normal_price = st.number_input("정상 요금(원/kWh)", value=288)
    daily_kwh = st.number_input("일평균 충전량(kWh/기)", value=20.0, step=1.0)
    
    st.markdown("---")
    st.write("**비용 설정**")
    kepco_base = st.number_input("한전 기본료(원/kW)", value=2390)
    kwh_cost = st.number_input("전력 매입단가(원/kWh)", value=150)
    monthly_maint = st.number_input("월 관리비(원/기)", value=10000)
    
    discount_rate_annual = st.slider("연 할인율(%) - NPV/IRR용", 1.0, 15.0, 5.0) / 100.0

# ==========================================
# [계산 로직: 월별(Monthly)]
# ==========================================

# 초기 투자비 계산
total_setup = (infra_cost + charger_cost) * num_units
total_subsidy = subsidy * num_units
net_capex = total_setup - total_subsidy
company_initial_outlay = net_capex - investment_amount

# 현금흐름 배열 초기화 (0시점 = 투자시점)
schedule = []
investor_cf = [-investment_amount] 
company_cf = [-company_initial_outlay]

# 상수 계산
avg_days_in_month = 365 / 12  
p1_end_month = p1_years * 12
p2_end_month = (p1_years + p2_years) * 12

# 월별 시뮬레이션 루프
for month_idx in range(1, total_months + 1):
    current_year = (month_idx - 1) // 12 + 1
    current_month_in_year = (month_idx - 1) % 12 + 1
    
    # A. 매출 계산
    is_promo = month_idx <= promo_months
    current_price = promo_price if is_promo else normal_price
    
    monthly_volume = daily_kwh * avg_days_in_month * num_units
    revenue = monthly_volume * current_price
        
    # B. 비용 계산
    base_cost = 7 * kepco_base * num_units 
    var_cost = monthly_volume * kwh_cost
    maint_cost = monthly_maint * num_units
    total_opex = base_cost + var_cost + maint_cost
    
    # C. 영업이익
    op_profit = revenue - total_opex
    
    # D. 운영 수익 배분
    op_investor_share = 0
    op_company_share = 0
    phase_label = ""
    
    if month_idx <= p1_end_month:
        phase_label = "1단계(이자)"
        op_investor_share = investment_amount * (p1_rate_annual / 12)
        op_company_share = op_profit - op_investor_share
    elif month_idx <= p2_end_month:
        phase_label = "2단계(배분)"
        if op_profit > 0:
            op_investor_share = op_profit * p2_share
            op_company_share = op_profit - op_investor_share
        else:
            op_investor_share = 0
            op_company_share = op_profit
    else:
        phase_label = "3단계(독점)"
        op_investor_share = 0
        op_company_share = op_profit
    
    # E. 원금 상환 로직
    principal_flow = 0
    if use_repayment and month_idx == repayment_month_idx:
        principal_flow = investment_amount
        phase_label += " (💰원금상환)"
    
    # 최종 현금흐름
    final_investor_flow = op_investor_share + principal_flow
    final_company_flow = op_company_share - principal_flow
        
    # 데이터 저장
    schedule.append({
        "누적월": month_idx,
        "년차": current_year,
        "월": current_month_in_year,
        "구분": phase_label,
        "매출": revenue,
        "비용(OPEX)": total_opex,
        "영업이익": op_profit,
        "투자자수익": final_investor_flow,
        "회사수익": final_company_flow
    })
    
    investor_cf.append(final_investor_flow)
    company_cf.append(final_company_flow)

# DataFrame 생성
df = pd.DataFrame(schedule)

# 누적 현금흐름(잔고) 계산
df["회사_누적현금"] = df["회사수익"].cumsum() - company_initial_outlay
df["Zero"] = 0 

# ==========================================
# [지표 계산 함수]
# ==========================================
def calculate_financials_monthly(monthly_cf, initial_investment, annual_discount_rate):
    monthly_rate = annual_discount_rate / 12
    npv = npf.npv(monthly_rate, monthly_cf)
    try:
        monthly_irr = npf.irr(monthly_cf)
        if pd.isna(monthly_irr): 
            annual_irr = 0
        else:
            annual_irr = (1 + monthly_irr) ** 12 - 1
    except:
        annual_irr = 0
        
    total_net_profit = sum(monthly_cf) 
    if initial_investment > 0:
        roi = (total_net_profit / initial_investment) * 100 
    else:
        roi = 0 
    return npv, annual_irr, roi

# 지표 계산
inv_npv, inv_irr, inv_roi = calculate_financials_monthly(investor_cf, investment_amount, discount_rate_annual)
com_npv, com_irr, com_roi = calculate_financials_monthly(company_cf, company_initial_outlay, discount_rate_annual)

# ==========================================
# [메인 화면 출력]
# ==========================================

st.subheader(f"📊 월별 정밀 분석 결과 ({simulation_years}년 / {total_months}개월)")
if use_repayment:
    st.info(f"💡 **상환 시점:** {repayment_year}년차 12월 ({repayment_month_idx}개월 차)에 원금 상환")

st.markdown("---")

# 1. 투자 분석 결과
st.subheader("💰 투자자 vs 회사 수익성 비교 (연환산 기준)")
col_inv, col_com = st.columns(2)

with col_inv:
    st.markdown("### 🧑‍💼 투자자 (Investor)")
    st.write(f"**투자액: {investment_amount:,.0f} 원**")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db;">
        <h2 style="margin:0; color: #0068c9;">ROI: {inv_roi:.1f} %</h2>
        <p style="margin:0;">연 IRR: {inv_irr*100:.2f} % | NPV: {inv_npv:,.0f} 원</p>
    </div>
    """, unsafe_allow_html=True)

with col_com:
    st.markdown("### 🏢 태성콘텍 (Company)")
    st.write(f"**초기 투입분: {company_initial_outlay:,.0f} 원**")
    com_roi_str = f"{com_roi:.1f} %" if company_initial_outlay > 0 else "N/A"
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #d1d5db;">
        <h2 style="margin:0; color: #2e7d32;">ROI: {com_roi_str}</h2>
        <p style="margin:0;">연 IRR: {com_irr*100:.2f} % | NPV: {com_npv:,.0f} 원</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 2. 시각화 (Altair 그래프)
st.subheader("📈 태성콘텍 현금흐름 분석 (Cash Flow & Balance)")

base = alt.Chart(df).encode(x=alt.X('누적월:Q', title='경과 월 (Month)'))

# [레이어 1] 누적 잔고 (좌측 Y축)
balance_line = base.mark_line(color='#2e7d32', strokeWidth=3).encode(
    y=alt.Y('회사_누적현금:Q', axis=alt.Axis(title='누적 현금 잔고 (원)', titleColor='#2e7d32')),
    tooltip=[alt.Tooltip('누적월'), alt.Tooltip('회사_누적현금', format=',.0f')]
)

balance_area = base.mark_area(opacity=0.1, color='#2e7d32').encode(
    y='회사_누적현금:Q'
)

# 0원 기준선
zero_rule = base.mark_rule(color='red', strokeDash=[5, 5]).encode(y='Zero:Q')

# [레이어 2] 월별 순수익 (우측 Y축)
monthly_bar = base.mark_bar(opacity=0.3, color='#1f77b4').encode(
    y=alt.Y('회사수익:Q', axis=alt.Axis(title='월별 순수익 (원)', titleColor='#1f77b4')),
    tooltip=[alt.Tooltip('누적월'), alt.Tooltip('회사수익', format=',.0f', title='월 순수익')]
)

# 차트 결합
chart = alt.layer(
    balance_area + balance_line + zero_rule, 
    monthly_bar                              
).resolve_scale(
    y='independent' 
).properties(
    height=400,
    title="월별 수익(막대) 및 누적 현금잔고(선) 복합 차트"
)

st.altair_chart(chart, use_container_width=True)

st.caption("""
**[그래프 보는 법]**
- **초록색 실선(좌측 축):** 태성콘텍의 통장 잔고입니다. 이 선이 0(빨간 점선) 위로 올라가야 원금 회수가 끝난 것입니다.
- **파란색 막대(우측 축):** 매달 들어오고 나가는 현금입니다. 원금을 상환하는 달에는 막대가 아래로 길게 내려갑니다.
""")


# 3. 상세 데이터 테이블
with st.expander("🗓️ 월별 상세 현금흐름표 (전체 보기)", expanded=False):
    cols_to_format = ["매출", "비용(OPEX)", "영업이익", "투자자수익", "회사수익", "회사_누적현금"]
    st.dataframe(
        df.style.format({col: "{:,.0f}" for col in cols_to_format}),
        use_container_width=True,
        height=400
    )

# CSV 다운로드
csv = df.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 월별 데이터 CSV 다운로드", csv, "ev_charging_monthly_roi.csv", "text/csv")