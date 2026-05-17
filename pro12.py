"""
태성콘텍 EV 충전인프라 수익성 분석 모델
Streamlit Web App

가정 반영:
  1) 1기당 월간비용: 기본료 18,060원 + 계시별 전력량요금 + 통신비 3,300원 + A/S비 5,000원 + 금융비용
  2) 1기당 월간수익: 일평균충전량 × 충전요금(정상 or 프로모션) × 30.416일
  3) 전력량요금은 건별 충전데이터 시간대 분포 × 한전 EV 계시별요금제로 산출
     프로모션: 6개월 or 1년, 136원 or 168원

실행: streamlit run pro12.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from datetime import datetime
import io


# ============================================================
# 전기자동차 충전전력요금 테이블 (충전서비스 제공사업자용)
# '26.6.1 시행 기준
# ============================================================

TARIFF_TABLE = {
    "저압-선택I": {
        "base_rate": 2390,
        "rates": {
            "summer":      {"경부하": 95.9,  "중간부하": 162.2, "최대부하": 203.5},
            "spring_fall": {"경부하": 85.4,  "중간부하": 97.2,  "최대부하": 102.1},
            "winter":      {"경부하": 110.6, "중간부하": 143.1, "최대부하": 172.0},
        },
    },
    "저압-선택II": {
        "base_rate": 2390,
        "rates": {
            "summer":      {"경부하": 83.1,  "중간부하": 140.0, "최대부하": 270.8},
            "spring_fall": {"경부하": 85.4,  "중간부하": 97.2,  "최대부하": 102.1},
            "winter":      {"경부하": 105.8, "중간부하": 126.7, "최대부하": 227.0},
        },
    },
    "저압-선택III": {
        "base_rate": 2390,
        "rates": {
            "summer":      {"경부하": 90.1,  "중간부하": 138.6, "최대부하": 236.0},
            "spring_fall": {"경부하": 85.4,  "중간부하": 97.2,  "최대부하": 102.1},
            "winter":      {"경부하": 115.5, "중간부하": 125.4, "최대부하": 198.4},
        },
    },
    "저압-선택IV": {
        "base_rate": 2390,
        "rates": {
            "summer":      {"경부하": 172.0, "중간부하": 172.0, "최대부하": 172.0},
            "spring_fall": {"경부하": 97.2,  "중간부하": 97.2,  "최대부하": 97.2},
            "winter":      {"경부하": 154.9, "중간부하": 154.9, "최대부하": 154.9},
        },
    },
    "고압-선택I": {
        "base_rate": 2580,
        "rates": {
            "summer":      {"경부하": 89.8,  "중간부하": 129.9, "최대부하": 151.2},
            "spring_fall": {"경부하": 80.2,  "중간부하": 91.0,  "최대부하": 94.9},
            "winter":      {"경부하": 99.4,  "중간부하": 118.4, "최대부하": 132.4},
        },
    },
    "고압-선택II": {
        "base_rate": 2580,
        "rates": {
            "summer":      {"경부하": 78.2,  "중간부하": 113.0, "최대부하": 198.6},
            "spring_fall": {"경부하": 80.2,  "중간부하": 91.0,  "최대부하": 94.9},
            "winter":      {"경부하": 95.2,  "중간부하": 105.5, "최대부하": 172.4},
        },
    },
    "고압-선택III": {
        "base_rate": 2580,
        "rates": {
            "summer":      {"경부하": 84.5,  "중간부하": 111.9, "최대부하": 174.0},
            "spring_fall": {"경부하": 80.2,  "중간부하": 91.0,  "최대부하": 94.9},
            "winter":      {"경부하": 103.6, "중간부하": 104.5, "최대부하": 151.6},
        },
    },
    "고압-선택IV": {
        "base_rate": 2580,
        "rates": {
            "summer":      {"경부하": 137.4, "중간부하": 137.4, "최대부하": 137.4},
            "spring_fall": {"경부하": 91.0,  "중간부하": 91.0,  "최대부하": 91.0},
            "winter":      {"경부하": 127.7, "중간부하": 127.7, "최대부하": 127.7},
        },
    },
}

TARIFF_NAMES = list(TARIFF_TABLE.keys())


def get_season(cal_month):
    if cal_month in (6, 7, 8):
        return "summer"
    elif cal_month in (3, 4, 5, 9, 10):
        return "spring_fall"
    return "winter"


SEASON_KR = {"summer": "여름철", "spring_fall": "봄·가을철", "winter": "겨울철"}
SEASON_MONTHS = {"summer": 3, "spring_fall": 5, "winter": 4}


def classify_hour_by_season(h, season):
    """계절별 시간대 분류 (한전 기준)
    여름·봄가을: 경부하 22~08, 중간부하 08~15·21~22, 최대부하 15~21
    겨울:       경부하 22~08, 중간부하 08~09·12~16·19~22, 최대부하 09~12·16~19
    """
    # 경부하: 모든 계절 동일 22:00~08:00
    if h >= 22 or h < 8:
        return "경부하"

    if season == "winter":
        if h in (8,) or 12 <= h < 16 or 19 <= h < 22:
            return "중간부하"
        else:  # 9~12, 16~19
            return "최대부하"
    else:  # summer, spring_fall
        if 8 <= h < 15 or h == 21:
            return "중간부하"
        else:  # 15~21
            return "최대부하"


def calc_seasonal_weighted_rate(tariff_key, hourly_kwh):
    """건별 데이터 시간대 분포 × 계절별 요금으로 연평균 가중 전력량요금 산출.
    반환: 연평균 원/kWh, 계절별 원/kWh dict, 계시별 비중 dict
    """
    tariff = TARIFF_TABLE[tariff_key]
    total_kwh = hourly_kwh.sum()
    if total_kwh == 0:
        return 0, {}, {}

    season_rates = {}
    for season, s_label in SEASON_KR.items():
        rates = tariff["rates"][season]
        # 시간대별 kWh를 해당 계절의 시간대 분류로 가중
        cost = 0.0
        for h in range(24):
            load_type = classify_hour_by_season(h, season)
            cost += hourly_kwh.get(h, 0) * rates[load_type]
        season_rates[season] = cost / total_kwh  # 원/kWh

    # 연평균 = 계절별 개월수 가중
    annual_avg = sum(season_rates[s] * SEASON_MONTHS[s] for s in season_rates) / 12

    # 계시별 비중 (봄가을 기준으로 표시용)
    load_split = {"경부하": 0, "중간부하": 0, "최대부하": 0}
    for h in range(24):
        lt = classify_hour_by_season(h, "spring_fall")
        load_split[lt] += hourly_kwh.get(h, 0)
    total = sum(load_split.values())
    load_pct = {k: v / total if total > 0 else 0 for k, v in load_split.items()}

    return annual_avg, season_rates, load_pct


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="태성콘텍 EV 충전 수익성 분석",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 비밀번호 보호
# ============================================================

def check_password():
    """비밀번호 입력 후 통과해야 앱 사용 가능"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 로그인")
    password = st.text_input("비밀번호를 입력하세요", type="password")

    if st.button("로그인", type="primary"):
        # secrets.toml 에 설정된 비밀번호와 비교 (없으면 기본값 사용)
        try:
            correct = st.secrets["app_password"]
        except (KeyError, FileNotFoundError):
            correct = "tsct2026"  # 기본 비밀번호 (secrets 미설정 시)

        if password == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            if password:
                st.error("비밀번호가 틀렸습니다.")
    return False


if not check_password():
    st.stop()

st.title("⚡ EV 충전인프라 수익성 분석 모델")
st.caption("환경부 보조금 / 자투자 / 위탁운영 시나리오 비교 · 10년 월별 손익")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ 분석 가정")

    # ---- 충전요금 (수익) ----
    st.subheader("💰 충전요금 (원/kWh)")
    price_normal = st.number_input("정상 충전요금", value=294.0, step=10.0, format="%.1f")
    st.caption("프로모션 요금·경과개월·계약기간은 아래 충전소별 테이블에서 개별 설정")

    # ---- 한전요금 ----
    st.subheader("⚡ 한전 전기요금")
    kepco_base = st.number_input("기본료 (원/기/월)", value=18_060, step=500)
    tariff_key = st.selectbox("계시별 요금제", TARIFF_NAMES, index=1,
                              help="전기자동차 충전전력요금(충전서비스 제공사업자용)")
    selected_tariff = TARIFF_TABLE[tariff_key]
    st.caption(f"요금제 기본요금: {selected_tariff['base_rate']:,}원/kW (참고, 위 기본료와 별도)")

    with st.expander("요금표 미리보기"):
        preview = []
        for s, s_label in SEASON_KR.items():
            r = selected_tariff["rates"][s]
            preview.append({"계절": s_label, "경부하": r["경부하"], "중간부하": r["중간부하"], "최대부하": r["최대부하"]})
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    # ---- 운영비 ----
    st.subheader("🔧 운영비 (원/기/월)")
    comm_fee = st.number_input("통신비", value=3_300, step=100)
    as_fee = st.number_input("A/S비", value=5_000, step=100)
    loan_rate_yr = st.number_input("금융비용 연이율 (%)", value=8.0, step=0.5) / 100

    # ---- CAPEX ----
    st.subheader("💵 CAPEX 시나리오 (1기당)")
    capex_unit = st.number_input("충전기 표준단가 (원)", value=3_000_000, step=100_000, format="%d")
    subsidy_ratio = st.number_input("보조금 비율 (%)", value=50.0, step=5.0) / 100
    capex_b = st.number_input("자투자 단가 (원)", value=2_300_000, step=100_000, format="%d")
    capex_c = st.number_input("위탁운영 단가 (원)", value=1_300_000, step=100_000, format="%d")

    # ---- 충전량 성장률 (EV 보급 + 차충비 연동) ----
    st.subheader("📈 충전량 성장률")
    st.caption("EV 보급 배수에서 충전기 증가분(차충비)을 차감 → 1기당 순 성장률 역산")

    growth_years = st.number_input(
        "성장 도달기간 (년)", value=5, step=1, min_value=1,
        help="목표 달성까지 기간 (예: 2026→2030 = 5년 아님 4년)")
    growth_period = growth_years * 12

    st.markdown("**EV 보급 전망**")
    ev_growth = st.number_input(
        "EV보급 목표 배수 (vs 현재)", value=3.0, step=0.5,
        help="예: 3.0 → EV 등록대수가 현재의 3배")

    st.markdown("**완속 충전기 증가 전망 (차충비)**")
    charger_now = st.number_input(
        "현재 전국 완속충전기 (만대)", value=50.0, step=5.0, format="%.1f")
    charger_target = st.number_input(
        "목표 전국 완속충전기 (만대)", value=120.0, step=5.0, format="%.1f",
        help="환경부 2030 목표")
    charger_growth = charger_target / charger_now if charger_now > 0 else 1.0

    # 1기당 순 성장 = EV 성장 / 충전기 성장
    net_growth = ev_growth / charger_growth if charger_growth > 0 else ev_growth

    # 월 성장률 역산
    if net_growth > 0 and growth_period > 0:
        g_phase1_pct = (net_growth ** (1 / growth_period) - 1) * 100
        g_annual_phase1 = ((1 + g_phase1_pct / 100) ** 12 - 1) * 100
    else:
        g_phase1_pct = 0
        g_annual_phase1 = 0

    st.info(
        f"📊 EV **{ev_growth:.1f}배** ÷ 충전기 **{charger_growth:.2f}배** "
        f"= 1기당 순 성장 **{net_growth:.2f}배**\n\n"
        f"→ 월 **{g_phase1_pct:.3f}%** (연 {g_annual_phase1:.1f}%) × {growth_years}년"
    )

    # 실제 적용되는 growth_target = 차충비 보정 후 순 성장
    growth_target = net_growth

    growth_post_pct = st.number_input(
        "목표 도달 이후 월 성장률 (%)", value=0.5, step=0.1,
        help="EV 보급 성숙기 이후 완만한 성장률")

    st.subheader("💹 요금 인상률")
    cpi_pct = st.number_input("충전요금 연 인상률 CPI (%)", value=2.5, step=0.5)
    kepco_inf_pct = st.number_input("한전요금 연 인상률 (%)", value=3.0, step=0.5)
    growth_post = growth_post_pct / 100
    cpi = cpi_pct / 100
    kepco_inflation = kepco_inf_pct / 100

    # ---- 재무 ----
    st.subheader("📊 재무 가정")
    depreciation_years = st.number_input("감가상각 기간 (년)", value=7, step=1)
    analysis_years = st.number_input("분석 기간 (년)", value=10, step=1)
    tax_rate = st.number_input("법인세 실효세율 (%, 지방세 포함)", value=22.0, step=1.0) / 100
    discount_rate = st.number_input("NPV 할인율 (%)", value=8.0, step=0.5) / 100
    replace_trigger = st.number_input("교체 트리거 (일평균 kWh/기)", value=30, step=5)


# ============================================================
# 파일 업로드
# ============================================================

st.header("📂 데이터 업로드")
st.markdown("""
### 📋 모델 가정 요약

| 항목 | 내용 |
|---|---|
| **수익** | 일평균충전량 × 충전요금 × 30.416일/월 |
| **프로모션** | 6개월 or 1년간 136원 or 168원 적용 가능 |
| **전력량요금** | 건별 충전 데이터 시간대 분포 × 한전 EV 계시별 요금제 |
| **기본료** | 18,060원/기/월 |
| **CAPEX** | A. 보조금(150만) / B. 자투자(230만) / C. 위탁(130만) |

아래 2개의 엑셀 파일을 업로드하면 분석이 시작됩니다.
""")

upload_col1, upload_col2 = st.columns(2)
with upload_col1:
    file_trans = st.file_uploader(
        "1. 건별 충전 데이터 (.xlsx)",
        type=["xlsx"],
        help="'충전량(kWh)', '충전시작' 컬럼 필요 → 시간대별 충전 패턴 추출",
    )
    if file_trans:
        st.success("건별 충전 데이터 업로드 완료")
    else:
        st.info("시간대별 충전 패턴 추출용 엑셀 파일")

with upload_col2:
    file_settle = st.file_uploader(
        "2. 가정산 데이터 (.xlsx)",
        type=["xlsx"],
        help="충전소별 가동률·매출 분배 비율 추출용",
    )
    if file_settle:
        st.success("가정산 데이터 업로드 완료")
    else:
        st.info("충전소별 가동률·매출 추출용 엑셀 파일")

if file_trans is None or file_settle is None:
    st.divider()
    st.warning("위 2개의 파일을 모두 업로드하면 분석이 자동으로 시작됩니다.")
    st.stop()


# ============================================================
# Load & clean data
# ============================================================

@st.cache_data
def load_data(_file_trans, _file_settle):
    df_t = pd.read_excel(_file_trans, dtype=str)

    def clean(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lstrip("'")
        if s in ("null", "", "nan"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    df_t["kWh"] = df_t["충전량(kWh)"].apply(clean)
    df_t["시작시간"] = pd.to_datetime(df_t["충전시작"], errors="coerce")
    df_t["hour"] = df_t["시작시간"].dt.hour

    df_s = pd.read_excel(_file_settle)
    return df_t, df_s


df_trans, df_settle_raw = load_data(file_trans, file_settle)


# ============================================================
# 0. 비가동 충전소 필터링
# ============================================================

st.header("🔍 충전소 필터링 (비가동 제외)")

min_kwh_threshold = st.slider(
    "최소 일평균 충전량 기준 (kWh/충전소) — 이 값 이하인 충전소는 제외",
    min_value=0.0, max_value=50.0, value=1.0, step=0.5,
)

n_all = len(df_settle_raw)
n_chargers_all = int(df_settle_raw["충전기수"].sum())

df_settle = df_settle_raw[df_settle_raw["일별평균충전량(kWh)"] > min_kwh_threshold].copy()

n_filtered = n_all - len(df_settle)
n_chargers_filtered = n_chargers_all - int(df_settle["충전기수"].sum())

fc1, fc2, fc3 = st.columns(3)
fc1.metric("전체 충전소", f"{n_all} 개소 / {n_chargers_all:,} 기")
fc2.metric("제외된 충전소 (비가동)", f"{n_filtered} 개소 / {n_chargers_filtered:,} 기",
           delta=f"-{n_filtered}" if n_filtered > 0 else "0", delta_color="inverse")
fc3.metric("분석 대상", f"{len(df_settle)} 개소 / {int(df_settle['충전기수'].sum()):,} 기")

if n_filtered > 0:
    with st.expander(f"제외된 충전소 목록 ({n_filtered}개)"):
        df_excluded = df_settle_raw[df_settle_raw["일별평균충전량(kWh)"] <= min_kwh_threshold].copy()
        show_cols = [c for c in ["충전소ID", "충전소명", "충전기수", "일별평균충전량(kWh)"] if c in df_excluded.columns]
        st.dataframe(
            df_excluded[show_cols].sort_values("일별평균충전량(kWh)").reset_index(drop=True),
            use_container_width=True,
        )

if len(df_settle) == 0:
    st.error("모든 충전소가 필터링되었습니다. 기준값을 낮춰주세요.")
    st.stop()

st.divider()


# ============================================================
# 1. 데이터 요약
# ============================================================

st.header("1️⃣ 데이터 요약 (가동 충전소 기준)")

n_stations = len(df_settle)
n_chargers = int(df_settle["충전기수"].sum())
total_kw = int(df_settle["총 전력"].sum())
# 일별평균충전량(kWh)은 이미 1기당 값 → 충전소 전체 = 1기당 × 충전기수
df_settle["기당_일평균"] = df_settle["일별평균충전량(kWh)"]  # 이미 1기당
df_settle["충전소_일평균"] = df_settle["일별평균충전량(kWh)"] * df_settle["충전기수"]  # 충전소 전체
daily_kwh = df_settle["충전소_일평균"].sum()  # 전체 합계

# 충전기수 가중평균 일평균충전량 (유효 충전기만)
# = Σ(1기당 일평균 × 충전기수) / Σ(충전기수) = daily_kwh / n_chargers
weighted_avg_kwh = daily_kwh / n_chargers if n_chargers > 0 else 0
util_pct = weighted_avg_kwh / (7 * 24) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("충전소 수", f"{n_stations} 개소")
c2.metric("유효 충전기수", f"{n_chargers:,} 기")
c3.metric("총 계약전력", f"{total_kw:,} kW")
c4.metric("전체 일평균 충전량", f"{daily_kwh:,.0f} kWh")
c5.metric("1기당 가중평균", f"{weighted_avg_kwh:.2f} kWh/기/일",
          help="충전기 설치대수를 가중치로 한 1기당 일평균충전량 (비가동 제외)")

st.caption(f"📊 가동률 약 **{util_pct:.1f}%** (7kW × 24h 기준) · "
           f"가중평균 = Σ(1기당 일평균 × 충전기수) ÷ 유효 충전기수 {n_chargers}기")

with st.expander("📊 충전소별 가동률 분포"):
    fig = px.histogram(
        df_settle, x="기당_일평균", nbins=30,
        title="충전기 1기당 일평균 충전량 분포",
        labels={"기당_일평균": "일평균 충전량 (kWh/기/일)"},
    )
    fig.add_vline(x=replace_trigger, line_dash="dash", line_color="red",
                  annotation_text=f"교체 트리거 ({replace_trigger}kWh)")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. 시간대별 충전 분포 → 계시별 전력량요금 산출
# ============================================================

st.header("2️⃣ 시간대별 충전 분포 & 계시별 전력량요금")

df_v = df_trans[df_trans["kWh"].notna() & (df_trans["kWh"] > 0)].copy()
hourly = df_v.groupby("hour")["kWh"].sum().reindex(range(24), fill_value=0)

# 계절별 가중평균 전력량요금 산출
annual_avg_rate, season_rates, load_pct = calc_seasonal_weighted_rate(tariff_key, hourly)

c1, c2 = st.columns([3, 1])
with c1:
    color_map = {
        h: "#4A90E2" if classify_hour_by_season(h, "spring_fall") == "경부하"
        else "#F5A623" if classify_hour_by_season(h, "spring_fall") == "중간부하"
        else "#D0021B"
        for h in range(24)
    }
    fig = go.Figure()
    fig.add_bar(
        x=list(range(24)),
        y=hourly.values,
        marker_color=[color_map[h] for h in range(24)],
        hovertemplate="%{x}시: %{y:,.1f} kWh<extra></extra>",
    )
    fig.update_layout(
        title="시간대별 충전량 분포 (건별 데이터 기반, 봄·가을 시간대 색상)",
        xaxis_title="시각", yaxis_title="충전량 (kWh)",
        height=350, xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("**계시별 충전 비중**")
    st.metric("경부하 (22~08시)", f"{load_pct.get('경부하', 0)*100:.1f}%")
    st.metric("중간부하", f"{load_pct.get('중간부하', 0)*100:.1f}%")
    st.metric("최대부하", f"{load_pct.get('최대부하', 0)*100:.1f}%")
    st.divider()
    st.markdown("**연평균 전력량요금**")
    st.metric("가중평균", f"{annual_avg_rate:.1f} 원/kWh")

# 계절별 상세
with st.expander("📊 계절별 가중평균 전력량요금 상세"):
    season_detail = []
    for s, s_label in SEASON_KR.items():
        r = selected_tariff["rates"][s]
        season_detail.append({
            "계절": s_label,
            "경부하(원/kWh)": r["경부하"],
            "중간부하(원/kWh)": r["중간부하"],
            "최대부하(원/kWh)": r["최대부하"],
            "가중평균(원/kWh)": f"{season_rates.get(s, 0):.1f}",
            "적용 개월": SEASON_MONTHS[s],
        })
    season_detail.append({
        "계절": "📌 연평균",
        "경부하(원/kWh)": "",
        "중간부하(원/kWh)": "",
        "최대부하(원/kWh)": "",
        "가중평균(원/kWh)": f"{annual_avg_rate:.1f}",
        "적용 개월": 12,
    })
    st.dataframe(pd.DataFrame(season_detail), use_container_width=True, hide_index=True)

# ============================================================
# 2-1. 시간대 이동(Demand Shifting) 시뮬레이션
# ============================================================

st.subheader("🔄 시간대 이동 시뮬레이션 (Demand Shifting)")
st.caption(
    "고객 서비스(예약충전, 야간할인 등)로 충전 시간대를 경부하로 이동시킬 때의 "
    "전기요금 절감 효과와 고객 수익 분배를 분석합니다."
)

col_orig, col_arrow, col_target = st.columns([2, 1, 2])

with col_orig:
    st.markdown("**현재 비중 (데이터 기반)**")
    orig_off = load_pct.get("경부하", 0) * 100
    orig_mid = load_pct.get("중간부하", 0) * 100
    orig_peak = load_pct.get("최대부하", 0) * 100
    st.metric("경부하", f"{orig_off:.1f}%")
    st.metric("중간부하", f"{orig_mid:.1f}%")
    st.metric("최대부하", f"{orig_peak:.1f}%")

with col_arrow:
    st.markdown("")
    st.markdown("")
    st.markdown("### →")

with col_target:
    st.markdown("**목표 비중 (조정)**")
    adj_off = st.slider("경부하 목표 (%)", 0, 100, int(round(orig_off)),
                         key="adj_off", help="야간충전 유도로 경부하 비중 증가")
    adj_mid = st.slider("중간부하 목표 (%)", 0, 100 - adj_off, int(round(orig_mid)),
                         key="adj_mid")
    adj_peak = 100 - adj_off - adj_mid
    st.info(f"최대부하: **{adj_peak}%**")

# 조정된 비중
adj_load_pct = {"경부하": adj_off / 100, "중간부하": adj_mid / 100, "최대부하": adj_peak / 100}

# 조정 후 연평균 전력량요금 계산 (계절별 가중)
def calc_rate_from_pct(tariff_key, load_pct_dict):
    """비중 dict로 연평균 전력량요금 산출"""
    tariff = TARIFF_TABLE[tariff_key]
    s_rates = {}
    for season in SEASON_KR:
        rates = tariff["rates"][season]
        s_rates[season] = sum(rates[lt] * load_pct_dict[lt] for lt in rates)
    annual = sum(s_rates[s] * SEASON_MONTHS[s] for s in s_rates) / 12
    return annual, s_rates

adj_annual_rate, adj_season_rates = calc_rate_from_pct(tariff_key, adj_load_pct)

# 절감액 계산
saving_per_kwh = annual_avg_rate - adj_annual_rate  # 원/kWh 절감 (공급가액 기준 아직 VAT 포함)
saving_per_kwh_supply = saving_per_kwh / 1.1         # 공급가액 기준 절감

st.divider()

# 절감 효과 + 분배 비율
st.markdown("**전기요금 절감 효과 & 고객 분배**")
eff1, eff2, eff3, eff4 = st.columns(4)
eff1.metric("현재 연평균 요금", f"{annual_avg_rate:.1f} 원/kWh")
eff2.metric("조정 후 연평균 요금", f"{adj_annual_rate:.1f} 원/kWh")
eff3.metric("절감액", f"{saving_per_kwh:.1f} 원/kWh",
            delta=f"{saving_per_kwh:.1f}원 ↓" if saving_per_kwh > 0 else "0",
            delta_color="inverse" if saving_per_kwh > 0 else "off")
eff4.metric("절감액(공급가액)", f"{saving_per_kwh_supply:.1f} 원/kWh")

share_ratio = st.slider(
    "고객 분배 비율 (%) — 절감액 중 고객에게 돌려주는 비율",
    0, 100, 50, step=5,
    help="0%: 절감액 전액 사업자 수익 / 100%: 절감액 전액 고객에게 환원 (충전요금 할인)"
)

customer_share_per_kwh = saving_per_kwh_supply * share_ratio / 100   # 고객에게 환원 (공급가액)
company_saving_per_kwh = saving_per_kwh_supply - customer_share_per_kwh  # 사업자 순 절감 (공급가액)

DAYS_PER_MONTH = 30.416  # 365 / 12

# 월 기준 효과 추정 (전체 충전량 기준)
monthly_kwh_total = daily_kwh * DAYS_PER_MONTH
monthly_saving_total = saving_per_kwh_supply * monthly_kwh_total
monthly_customer_share = customer_share_per_kwh * monthly_kwh_total
monthly_company_gain = company_saving_per_kwh * monthly_kwh_total

sh1, sh2, sh3 = st.columns(3)
sh1.metric("월 총 절감액", f"{monthly_saving_total:,.0f}원",
           help=f"전체 {daily_kwh:,.0f}kWh/일 기준")
sh2.metric(f"고객 환원 ({share_ratio}%)", f"{monthly_customer_share:,.0f}원/월",
           help="충전요금 할인 등으로 고객에게 돌려줌")
sh3.metric(f"사업자 순이익 증가 ({100-share_ratio}%)", f"{monthly_company_gain:,.0f}원/월",
           help="시간대 이동으로 인한 사업자 추가 수익")

st.caption(
    f"📊 1기당 효과: 절감 {saving_per_kwh_supply:.1f}원/kWh → "
    f"고객 {customer_share_per_kwh:.1f}원 + 사업자 {company_saving_per_kwh:.1f}원/kWh"
)


# ============================================================
# 3. 충전소별 시나리오 지정 & 개소별 수익성 분석
# ============================================================

st.header("3️⃣ 충전소별 시나리오 지정 & 수익성 분석")

SCENARIO_OPTIONS = ["A. 보조금", "B. 자투자", "C. 위탁운영"]
CAPEX_MAP = {
    "A. 보조금": capex_unit * (1 - subsidy_ratio),
    "B. 자투자": capex_b,
    "C. 위탁운영": capex_c,
}

st.markdown(f"""
각 충전소(아파트)에 적용할 사업모델을 선택하세요.
| 시나리오 | 1기당 투자비 |
|---|---|
| **A. 보조금** | {CAPEX_MAP['A. 보조금']:,.0f}원 ({capex_unit:,.0f} × {(1-subsidy_ratio)*100:.0f}%) |
| **B. 자투자** | {capex_b:,.0f}원 |
| **C. 위탁운영** | {capex_c:,.0f}원 |
""")

# 충전소별 시나리오·프로모션·계약기간 선택 테이블 구성
name_col = "충전소명" if "충전소명" in df_settle.columns else "충전소ID"
assign_df = df_settle[[name_col, "충전기수", "일별평균충전량(kWh)", "기당_일평균"]].copy()

# 일괄 지정
st.markdown("**일괄 지정**")
bc1, bc2, bc3, bc4 = st.columns(4)
with bc1:
    bulk_scenario = st.selectbox("시나리오", ["개별 선택"] + SCENARIO_OPTIONS)
with bc2:
    bulk_price_type = st.selectbox("요금상태", ["개별 선택", "프로모션", "정상요금"])
with bc3:
    bulk_promo_rate = st.selectbox("프로모션 단가", ["개별 선택", 136, 168])
with bc4:
    bulk_contract = st.selectbox("계약기간", ["개별 선택", 5, 7, 10])

# 기본값 설정
assign_df["시나리오"] = bulk_scenario if bulk_scenario != "개별 선택" else "C. 위탁운영"
assign_df["요금상태"] = bulk_price_type if bulk_price_type != "개별 선택" else "프로모션"
assign_df["프로모션단가(원)"] = int(bulk_promo_rate) if bulk_promo_rate != "개별 선택" else 168
assign_df["프로모션경과(월)"] = 0  # 기본: 아직 시작 전
assign_df["프로모션기간(월)"] = 12
assign_df["계약기간(년)"] = int(bulk_contract) if bulk_contract != "개별 선택" else 10

st.caption("아래 테이블에서 충전소별로 개별 수정 가능합니다.")

assign_edited = st.data_editor(
    assign_df,
    use_container_width=True,
    hide_index=True,
    disabled=[name_col, "충전기수", "일별평균충전량(kWh)", "기당_일평균"],
    column_config={
        "시나리오": st.column_config.SelectboxColumn(
            "시나리오", options=SCENARIO_OPTIONS, required=True,
        ),
        "요금상태": st.column_config.SelectboxColumn(
            "요금상태", options=["프로모션", "정상요금"], required=True,
            help="현재 프로모션 적용 중인지 여부",
        ),
        "프로모션단가(원)": st.column_config.SelectboxColumn(
            "프로모션단가", options=[136, 168], required=True,
        ),
        "프로모션경과(월)": st.column_config.NumberColumn(
            "프로모션경과(월)", format="%d", min_value=0, max_value=24,
            help="프로모션 시작 후 경과 개월수 (0=이번달 시작)",
        ),
        "프로모션기간(월)": st.column_config.SelectboxColumn(
            "프로모션기간", options=[6, 12], required=True,
            help="프로모션 총 적용 기간",
        ),
        "계약기간(년)": st.column_config.SelectboxColumn(
            "계약기간(년)", options=[3, 5, 7, 10], required=True,
        ),
        "충전기수": st.column_config.NumberColumn("충전기수", format="%d"),
        "일별평균충전량(kWh)": st.column_config.NumberColumn("1기당 일평균", format="%.1f"),
        "기당_일평균": st.column_config.NumberColumn("기당_일평균(kWh)", format="%.2f"),
    },
    key="assign_table",
)

# 시나리오별 요약
st.subheader("시나리오별 배정 현황")
sc1, sc2, sc3 = st.columns(3)
for col, sc_name in zip([sc1, sc2, sc3], SCENARIO_OPTIONS):
    mask = assign_edited["시나리오"] == sc_name
    n_st = mask.sum()
    n_ch = int(assign_edited.loc[mask, "충전기수"].sum())
    col.metric(sc_name, f"{n_st}개소 / {n_ch}기",
               help=f"1기당 {CAPEX_MAP[sc_name]:,.0f}원 → 총 {CAPEX_MAP[sc_name]*n_ch:,.0f}원")

st.divider()

# ============================================================
# 공통 계산 변수
# ============================================================

# DAYS_PER_MONTH는 상단에서 정의됨
n_months = analysis_years * 12

g_phase1 = growth_target ** (1 / growth_period) - 1
growth_mult = np.zeros(n_months)
growth_mult[0] = 1.0
for m in range(1, n_months):
    if m < growth_period:
        growth_mult[m] = growth_mult[m - 1] * (1 + g_phase1)
    else:
        growth_mult[m] = growth_mult[m - 1] * (1 + growth_post)

price_mult = np.array([(1 + cpi) ** (m / 12) for m in range(n_months)])
kepco_mult = np.array([(1 + kepco_inflation) ** (m / 12) for m in range(n_months)])

total_kwh_data = df_settle["전체충전량[kWh]"].sum()


def calc_station(station_n_chargers, station_daily_kwh, capex_per_unit,
                 is_promo, promo_unit_price, promo_elapsed, promo_total_months,
                 contract_years_station):
    """개별 충전소 단위 월별 시뮬레이션 (충전소별 프로모션·계약기간)

    is_promo: 프로모션 적용 중 여부
    promo_unit_price: 프로모션 단가 (원/kWh)
    promo_elapsed: 이미 경과한 프로모션 개월수
    promo_total_months: 프로모션 총 기간 (6 or 12)
    contract_years_station: 이 충전소의 계약기간 (년)
    """
    station_months = contract_years_station * 12
    total_capex_gross = capex_per_unit * station_n_chargers  # CAPEX (VAT 포함 가정)
    capex_vat_refund = total_capex_gross / 11                # CAPEX 매입세액 환급
    total_capex = total_capex_gross - capex_vat_refund       # 실질 투자비 (공급가액)
    monthly_dep = total_capex / (min(depreciation_years, contract_years_station) * 12)
    monthly_finance = total_capex * loan_rate_yr / 12
    monthly_kwh_base = station_daily_kwh * DAYS_PER_MONTH

    # 프로모션 잔여 개월
    promo_remaining = max(0, promo_total_months - promo_elapsed) if is_promo else 0

    records = []
    # 초기 현금유출 = 총 CAPEX 지출 - VAT 환급 = 공급가액
    cum_cf = -total_capex
    start = datetime(2026, 5, 1)

    for m in range(station_months):
        kwh_m = monthly_kwh_base * (growth_mult[m] if m < len(growth_mult) else growth_mult[-1])

        if m < promo_remaining:
            charge_price = float(promo_unit_price)
        else:
            charge_price = price_normal * (price_mult[m] if m < len(price_mult) else price_mult[-1])
        # ========== 부가세 처리 (공급가액 기준 모델링) ==========
        # 매출: 고객 결제액(VAT 포함) → 공급가액만 실질 매출
        revenue_gross = kwh_m * charge_price        # 고객 결제액 (VAT 포함)
        vat_sales = revenue_gross / 11              # 매출세액 (10/110)
        revenue = revenue_gross - vat_sales         # 공급가액 = 실질 매출

        # 비용: 한전 요금도 VAT 포함 금액 → 공급가액만 원가 반영
        km = kepco_mult[m] if m < len(kepco_mult) else kepco_mult[-1]
        kepco_base_gross = station_n_chargers * kepco_base * km   # VAT 포함
        cal_month = ((start.month - 1 + m) % 12) + 1
        season = get_season(cal_month)
        s_rate = season_rates.get(season, annual_avg_rate)
        kepco_var_gross = kwh_m * s_rate * km                     # VAT 포함

        kepco_gross = kepco_base_gross + kepco_var_gross
        vat_kepco = kepco_gross / 11                # 전기요금 매입세액
        kepco_supply = kepco_gross - vat_kepco       # 전기요금 공급가액 (실질 원가)

        comm_cost = station_n_chargers * comm_fee
        as_cost = station_n_chargers * as_fee
        opex = kepco_supply + comm_cost + as_cost    # OPEX = 공급가액 기준

        # 부가세 납부액 = 매출세액 - 매입세액 (pass-through, 손익 외)
        vat_payable = vat_sales - vat_kepco

        # 손익 (공급가액 기준)
        ebitda = revenue - opex
        in_dep = m < depreciation_years * 12
        dep = monthly_dep if in_dep else 0
        ebit = ebitda - dep
        finance = monthly_finance if in_dep else 0
        pretax = ebit - finance
        tax = max(0, pretax) * tax_rate
        ni = pretax - tax

        # 현금흐름 = 순이익 + 감가(비현금) - 부가세 납부액(pass-through)
        cf = ni + dep - vat_payable
        cum_cf += cf

        records.append({
            "월차": m + 1,
            "연월": start + relativedelta(months=m),
            "충전량_kWh": kwh_m,
            "충전요금_단가": charge_price,
            "전력량요금_단가": s_rate * km,
            "고객결제액": revenue_gross,
            "매출세액": vat_sales,
            "매출(공급가액)": revenue,
            "한전(공급가액)": kepco_supply,
            "매입세액": vat_kepco,
            "통신비": comm_cost,
            "A/S비": as_cost,
            "영업비(공급가액)": opex,
            "EBITDA": ebitda,
            "감가상각": dep,
            "EBIT": ebit,
            "금융비용": finance,
            "세전이익": pretax,
            "법인세": tax,
            "당기순이익": ni,
            "VAT납부액": vat_payable,
            "월현금흐름": cf,
            "누적현금흐름": cum_cf,
        })

    df = pd.DataFrame(records)

    # NPV
    monthly_disc = (1 + discount_rate) ** (1 / 12) - 1
    discount_factors = np.array([(1 + monthly_disc) ** (i + 1) for i in range(station_months)])
    npv = -total_capex + sum(df["월현금흐름"].values / discount_factors)

    # IRR
    irr_annual = None
    try:
        from scipy.optimize import brentq
        cfs_irr = np.concatenate([[-total_capex], df["월현금흐름"].values])
        def npv_func(r):
            return sum(cfs_irr[t] / (1 + r) ** t for t in range(len(cfs_irr)))
        if npv_func(-0.99) * npv_func(1.0) < 0:
            irr_monthly = brentq(npv_func, -0.99, 1.0)
            irr_annual = (1 + irr_monthly) ** 12 - 1
    except Exception:
        pass

    # Payback
    payback = None
    for _, row in df.iterrows():
        if row["누적현금흐름"] >= 0:
            payback = int(row["월차"])
            break

    return {
        "df": df,
        "total_capex": total_capex,
        "npv": npv,
        "irr_annual": irr_annual,
        "payback": payback,
        "total_revenue": df["매출(공급가액)"].sum(),
        "total_opex": df["영업비(공급가액)"].sum(),
        "total_ni": df["당기순이익"].sum(),
        "total_cf": df["월현금흐름"].sum(),
        "final_cum_cf": df["누적현금흐름"].iloc[-1],
    }


# ============================================================
# 개소별 수익성 계산
# ============================================================

with st.spinner("충전소별 수익성 계산 중..."):
    station_results = []
    station_dfs = []  # 월별 데이터 보존 (합산용)

    for idx, row in assign_edited.iterrows():
        sc = row["시나리오"]
        capex_pu = CAPEX_MAP[sc]
        n_ch = int(row["충전기수"])
        per_charger_kwh = row["일별평균충전량(kWh)"]  # 이미 1기당
        station_total_kwh = per_charger_kwh * n_ch  # 충전소 전체 일평균
        sname = row[name_col]

        # 충전소별 프로모션·계약기간
        is_promo = (row["요금상태"] == "프로모션")
        promo_up = int(row["프로모션단가(원)"])
        promo_elapsed = int(row["프로모션경과(월)"])
        promo_total = int(row["프로모션기간(월)"])
        contract_yrs = int(row["계약기간(년)"])

        r = calc_station(
            n_ch, station_total_kwh, capex_pu,
            is_promo=is_promo,
            promo_unit_price=promo_up,
            promo_elapsed=promo_elapsed,
            promo_total_months=promo_total,
            contract_years_station=contract_yrs,
        )
        station_dfs.append(r["df"])

        promo_remaining = max(0, promo_total - promo_elapsed) if is_promo else 0
        station_results.append({
            "충전소": sname,
            "시나리오": sc,
            "충전기수": n_ch,
            "기당_일평균(kWh)": per_charger_kwh,
            "요금상태": row["요금상태"],
            "프로모션단가": promo_up if is_promo else "-",
            "프로모션잔여(월)": promo_remaining if is_promo else 0,
            "계약기간(년)": contract_yrs,
            "1기당 투자비(원)": capex_pu,
            "총 투자비(원)": r["total_capex"],
            f"매출(원)": r["total_revenue"],
            f"순이익(원)": r["total_ni"],
            "NPV(원)": r["npv"],
            "연 IRR": r["irr_annual"],
            "회수기간(월)": r["payback"],
            "최종 누적CF(원)": r["final_cum_cf"],
        })

    station_summary = pd.DataFrame(station_results)


# ============================================================
# 개소별 수익성 결과
# ============================================================

st.subheader("📊 충전소별 수익성 결과")

# 수익성 표시
fmt_dict = {
    "충전기수": "{:d}",
    "기당_일평균(kWh)": "{:.2f}",
    "프로모션잔여(월)": "{:d}",
    "계약기간(년)": "{:d}",
    "1기당 투자비(원)": "{:,.0f}",
    "총 투자비(원)": "{:,.0f}",
    "매출(원)": "{:,.0f}",
    "순이익(원)": "{:,.0f}",
    "NPV(원)": "{:,.0f}",
    "연 IRR": lambda v: f"{v*100:.2f}%" if pd.notna(v) and v is not None else "—",
    "회수기간(월)": lambda v: f"{int(v)}개월" if pd.notna(v) and v is not None else "미회수",
    "최종 누적CF(원)": "{:,.0f}",
}
st.dataframe(
    station_summary.style.format(fmt_dict).map(
        lambda v: "color: red" if isinstance(v, (int, float)) and v < 0 else "",
        subset=["순이익(원)", "NPV(원)", "최종 누적CF(원)"]
    ),
    use_container_width=True, height=400,
)

# 전체 합산
st.subheader("📋 전체 투자 종합 수익성")
total_capex_all = station_summary["총 투자비(원)"].sum()
total_rev_all = station_summary["매출(원)"].sum()
total_ni_all = station_summary["순이익(원)"].sum()
total_npv_all = station_summary["NPV(원)"].sum()
total_cf_all = station_summary["최종 누적CF(원)"].sum()
n_profitable = (station_summary["순이익(원)"] > 0).sum()

# --- 전체 합산 ROI / MOIC / IRR / 회수기간 ---
max_months = max(len(d) for d in station_dfs)
ref_start = datetime(2026, 5, 1)
# 월별 현금흐름 합산 (계약기간 다를 수 있으므로 패딩)
padded_cfs = []
for d in station_dfs:
    vals = d["월현금흐름"].values
    if len(vals) < max_months:
        vals = np.concatenate([vals, np.zeros(max_months - len(vals))])
    padded_cfs.append(vals)
all_monthly_cf = sum(padded_cfs)
all_cashflows = np.concatenate([[-total_capex_all], all_monthly_cf])

# ROI = 누적순현금흐름 / 총투자비 × 100
total_net_cf = all_monthly_cf.sum()
total_roi = total_net_cf / total_capex_all * 100 if total_capex_all > 0 else 0

# MOIC = (누적순현금흐름 + 총투자비) / 총투자비
total_moic = (total_net_cf + total_capex_all) / total_capex_all if total_capex_all > 0 else 0

# IRR (월별 → 연환산)
total_irr_annual = None
try:
    from scipy.optimize import brentq
    def _npv_total(r):
        return sum(all_cashflows[t] / (1 + r) ** t for t in range(len(all_cashflows)))
    if _npv_total(-0.99) * _npv_total(1.0) < 0:
        irr_m = brentq(_npv_total, -0.99, 1.0)
        total_irr_annual = (1 + irr_m) ** 12 - 1
except Exception:
    pass

# 회수기간
cum_cf_all = np.cumsum(all_cashflows)
total_payback = None
for i in range(1, len(cum_cf_all)):
    if cum_cf_all[i] >= 0 and cum_cf_all[i - 1] < 0:
        total_payback = i
        break

# KPI 카드
st.markdown("#### 핵심 투자지표")
ki1, ki2, ki3, ki4, ki5 = st.columns(5)
ki1.metric("총 투자비", f"{total_capex_all/1e8:.2f}억")
ki2.metric("ROI", f"{total_roi:,.1f}%", help="누적순현금흐름 / 총투자비 × 100")
ki3.metric("MOIC", f"{total_moic:,.2f}x", help="(누적CF + 투자비) / 투자비")
ki4.metric("IRR (연)", f"{total_irr_annual*100:.2f}%" if total_irr_annual is not None else "—",
           help="월별 CF 기반 연환산")
ki5.metric("회수기간", f"{total_payback}개월" if total_payback else "미회수")

st.markdown("#### 손익 요약")
tc1, tc2, tc3, tc4, tc5 = st.columns(5)
tc1.metric("총매출", f"{total_rev_all/1e8:.2f}억")
tc2.metric("총순이익", f"{total_ni_all/1e8:.2f}억")
tc3.metric("총 NPV", f"{total_npv_all/1e8:.2f}억")
tc4.metric("총 누적CF", f"{total_cf_all/1e8:.2f}억")
tc5.metric("흑자 충전소", f"{n_profitable} / {len(station_summary)} 개소")

# 시나리오별 소계
st.subheader("시나리오별 소계")
for sc_name in SCENARIO_OPTIONS:
    sc_mask = station_summary["시나리오"] == sc_name
    if sc_mask.sum() == 0:
        continue
    sc_sub = station_summary[sc_mask]
    sc_capex = sc_sub["총 투자비(원)"].sum()
    sc_cf = sc_sub["최종 누적CF(원)"].sum()
    sc_roi = sc_cf / sc_capex * 100 if sc_capex > 0 else 0
    sc_moic = (sc_cf + sc_capex) / sc_capex if sc_capex > 0 else 0
    with st.container(border=True):
        st.markdown(f"**{sc_name}** — {sc_mask.sum()}개소, {int(sc_sub['충전기수'].sum())}기")
        cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
        cc1.metric("총 투자비", f"{sc_capex:,.0f}원")
        cc2.metric("ROI", f"{sc_roi:,.1f}%")
        cc3.metric("MOIC", f"{sc_moic:,.2f}x")
        cc4.metric("순이익", f"{sc_sub['순이익(원)'].sum():,.0f}원")
        cc5.metric("총 NPV", f"{sc_sub['NPV(원)'].sum():,.0f}원")
        cc6.metric("흑자 비율", f"{(sc_sub['순이익(원)'] > 0).sum()}/{len(sc_sub)}")

st.divider()

# ============================================================
# 전체 합산 월별 추이 차트
# ============================================================

st.subheader("📈 전체 합산 월별 추이")

# 시나리오별 합산 월별 데이터
agg_by_scenario = {}
for sc_name in SCENARIO_OPTIONS:
    sc_mask = assign_edited["시나리오"] == sc_name
    if sc_mask.sum() == 0:
        continue
    sc_indices = sc_mask[sc_mask].index.tolist()
    sc_dfs = [station_dfs[i] for i in range(len(station_dfs)) if assign_edited.index[i] in sc_indices]
    if sc_dfs:
        sc_max = max(len(d) for d in sc_dfs)
        combined = pd.DataFrame({
            "연월": [ref_start + relativedelta(months=m) for m in range(sc_max)],
        })
        for col in ["매출(공급가액)", "영업비(공급가액)", "당기순이익", "EBITDA", "월현금흐름"]:
            padded = []
            for d in sc_dfs:
                vals = d[col].values
                if len(vals) < sc_max:
                    vals = np.concatenate([vals, np.zeros(sc_max - len(vals))])
                padded.append(vals)
            combined[col] = sum(padded)
        sc_total_capex = sum(
            CAPEX_MAP[sc_name] * int(assign_edited.loc[i, "충전기수"])
            for i in sc_indices
        )
        combined["누적현금흐름"] = combined["월현금흐름"].cumsum() - sc_total_capex
        agg_by_scenario[sc_name] = combined

# 전체 합산 (계약기간이 다를 수 있으므로 최대 길이 기준, 짧은 건 0 패딩)
all_combined = pd.DataFrame({
    "연월": [ref_start + relativedelta(months=m) for m in range(max_months)],
})
for col in ["매출(공급가액)", "영업비(공급가액)", "당기순이익", "EBITDA", "월현금흐름"]:
    padded = []
    for d in station_dfs:
        vals = d[col].values
        if len(vals) < max_months:
            vals = np.concatenate([vals, np.zeros(max_months - len(vals))])
        padded.append(vals)
    all_combined[col] = sum(padded)
all_combined["누적현금흐름"] = all_combined["월현금흐름"].cumsum() - total_capex_all

tab1, tab2, tab3 = st.tabs(["누적 현금흐름", "월별 매출 vs 영업비용", "월별 EBITDA"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_combined["연월"], y=all_combined["누적현금흐름"],
        mode="lines", name="전체 합산", line=dict(width=3, color="black"),
        hovertemplate="%{x|%Y-%m}: %{y:,.0f}원<extra>전체</extra>",
    ))
    colors = {"A. 보조금": "#2ECC71", "B. 자투자": "#3498DB", "C. 위탁운영": "#E67E22"}
    for sc_name, sc_df in agg_by_scenario.items():
        fig.add_trace(go.Scatter(
            x=sc_df["연월"], y=sc_df["누적현금흐름"],
            mode="lines", name=sc_name, line=dict(dash="dot", color=colors.get(sc_name, "gray")),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="회수 분기점")
    fig.update_layout(title="누적 현금흐름 (전체 + 시나리오별)", height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_combined["연월"], y=all_combined["매출(공급가액)"],
                             mode="lines", name="매출", fill="tozeroy", line=dict(color="#2ECC71")))
    fig.add_trace(go.Scatter(x=all_combined["연월"], y=all_combined["영업비(공급가액)"],
                             mode="lines", name="영업비용", line=dict(color="#E74C3C")))
    fig.update_layout(title="월별 매출·영업비용 (전체 합산)", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_combined["연월"], y=all_combined["EBITDA"],
                             mode="lines", name="EBITDA", line=dict(color="#9B59B6")))
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="EBITDA 흑전")
    fig.update_layout(title="월별 EBITDA (전체 합산)", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# 개소별 상세 드릴다운
with st.expander("🔍 개별 충전소 상세 월별 손익"):
    station_names = assign_edited[name_col].tolist()
    sel_station = st.selectbox("충전소 선택", station_names, key="station_drill")
    sel_idx = station_names.index(sel_station)
    sel_df = station_dfs[sel_idx].copy()
    sel_df["연월"] = sel_df["연월"].dt.strftime("%Y-%m")
    sel_sc = assign_edited.iloc[sel_idx]["시나리오"]
    sel_ch = int(assign_edited.iloc[sel_idx]["충전기수"])
    st.caption(f"시나리오: **{sel_sc}** | 충전기: **{sel_ch}기** | "
               f"1기당 투자비: **{CAPEX_MAP[sel_sc]:,.0f}원** | 총 투자: **{CAPEX_MAP[sel_sc]*sel_ch:,.0f}원**")

    key_cols = [
        "월차", "연월", "충전량_kWh", "충전요금_단가",
        "고객결제액", "매출세액", "매출(공급가액)",
        "한전(공급가액)", "매입세액", "통신비", "A/S비", "영업비(공급가액)",
        "EBITDA", "감가상각", "금융비용",
        "당기순이익", "VAT납부액", "월현금흐름", "누적현금흐름",
    ]
    st.dataframe(
        sel_df[key_cols].style.format({c: "{:,.0f}" for c in key_cols if c not in ("월차", "연월")}),
        use_container_width=True, height=350,
    )

# comp_df for Excel export
comp_df = station_summary.copy()


# ============================================================
# 4. 교체 후보
# ============================================================

st.header("4️⃣ 급속 교체 후보 (일평균 {0}kWh/기 초과)".format(replace_trigger))

df_replace = df_settle.sort_values("기당_일평균", ascending=False).copy()
candidates = df_replace[df_replace["기당_일평균"] >= replace_trigger]

c1, c2 = st.columns([1, 3])
with c1:
    st.metric("교체 후보 충전소", f"{len(candidates)} 개소")
    st.metric("교체 후보 충전기", f"{int(candidates['충전기수'].sum())} 기")
with c2:
    if len(candidates) > 0:
        st.dataframe(
            candidates[["충전소ID", "충전소명", "충전기수", "일별평균충전량(kWh)", "기당_일평균"]]
            .rename(columns={"기당_일평균": "기당_일평균_kWh"})
            .reset_index(drop=True),
            use_container_width=True, height=300,
        )
    else:
        st.info(f"일평균 {replace_trigger} kWh/기를 초과하는 충전소가 없습니다.")


# ============================================================
# 5. 원본 데이터 검증
# ============================================================

st.header("5️⃣ 원본 데이터 검증 (Sanity Check)")

days_in_data = total_kwh_data / daily_kwh if daily_kwh > 0 else 0
months_in_data = days_in_data / 30
original_avg_price = (
    df_settle["충전결제요금-전체[원]"].sum() / total_kwh_data if total_kwh_data > 0 else 0
)

monthly_rev_orig = df_settle["충전결제요금-전체[원]"].sum() / months_in_data if months_in_data > 0 else 0
monthly_rev_model = all_combined["매출(공급가액)"].iloc[0] if len(all_combined) > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("가정산 데이터 기간 (추정)", f"{days_in_data:.0f}일 (≈{months_in_data:.1f}개월)")
c2.metric("원본 평균 충전단가", f"{original_avg_price:.1f} 원/kWh")
c3.metric("모델 월매출 vs 원본 월매출",
          f"{monthly_rev_model/1e6:.1f}M",
          f"{(monthly_rev_model - monthly_rev_orig)/1e6:+.1f}M")

with st.expander("⚠️ 모델 가정 vs 원본 데이터 차이"):
    n_promo = (assign_edited["요금상태"] == "프로모션").sum()
    promo_info = f"프로모션 적용 중: {n_promo}개소 → " if n_promo > 0 else ""
    st.markdown(f"""
- **원본 데이터 평균 단가: {original_avg_price:.1f} 원/kWh** (매출/충전량)
- **모델 충전요금: {promo_info}정상 {price_normal} 원/kWh** (충전소별 프로모션 개별 적용)
- **계시별 전력량요금 연평균: {annual_avg_rate:.1f} 원/kWh** ({tariff_key})
  - 경부하 {load_pct.get('경부하',0)*100:.1f}% / 중간부하 {load_pct.get('중간부하',0)*100:.1f}% / 최대부하 {load_pct.get('최대부하',0)*100:.1f}%
- **월 일수: {DAYS_PER_MONTH}일** (365÷12)
    """)


# ============================================================
# 6. Excel 다운로드
# ============================================================

st.header("6️⃣ 결과 내보내기")


@st.cache_data
def to_excel(_station_summary, _all_combined, _station_dfs, _station_names):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _station_summary.to_excel(writer, sheet_name="충전소별_수익성", index=False)

        agg_out = _all_combined.copy()
        agg_out["연월"] = agg_out["연월"].dt.strftime("%Y-%m")
        agg_out.to_excel(writer, sheet_name="전체합산_월별", index=False)

        # 개별 충전소 상위 10개 (시트 수 제한)
        for i, sname in enumerate(_station_names[:10]):
            df_out = _station_dfs[i].copy()
            df_out["연월"] = df_out["연월"].dt.strftime("%Y-%m")
            sheet = f"충전소{i+1}"
            df_out.to_excel(writer, sheet_name=sheet, index=False)
    return output.getvalue()


station_names_list = assign_edited[name_col].tolist()
excel_bytes = to_excel(station_summary, all_combined, station_dfs, station_names_list)
st.download_button(
    "📥 분석 결과 다운로드 (Excel)",
    data=excel_bytes,
    file_name=f"태성콘텍_수익성분석_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption(
    "ⓘ 모든 가정값은 좌측 사이드바에서 실시간 조정 가능합니다. "
    f"전력량요금은 {tariff_key} 요금제 기준 계절별·시간대별 가중평균 적용. "
    f"월 일수 {DAYS_PER_MONTH}일(365/12). 변경 시 모든 결과가 즉시 재계산됩니다."
)
