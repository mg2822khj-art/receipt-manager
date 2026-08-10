import os
import sys
import json
import re
import io
from datetime import datetime, time as dtime

# Windows 한글 인코딩 강제 설정
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
import plotly.express as px
from google import genai
from google.genai import types

load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────
CSV_PATH = "receipts.csv"
CATEGORIES = ["식대", "판촉", "소모품비", "차량유지비", "기타"]
AUTHORS = ["대표님", "권혁제"]
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
당신은 대한민국 법인카드 영수증을 분석하는 최고 수준의 OCR 전문 AI입니다.
이미지에서 영수증 정보를 정확하게 추출하여 반드시 아래 JSON 형식으로만 응답하세요.

규칙:
1. 응답은 반드시 순수 JSON만 출력하세요. 설명, 마크다운 코드블록, 백틱 없이 JSON 객체만 출력.
2. date는 YYYY-MM-DD 형식으로 변환하세요. 날짜를 인식할 수 없으면 오늘 날짜를 사용하세요.
3. time은 HH:MM 형식(24시간)으로 추출하세요. 없으면 빈 문자열("")로 출력하세요.
4. amount는 부가세 포함 총 결제금액(정수)만 추출하세요. 콤마나 단위는 제거하고 숫자만.
5. merchant는 실제 가맹점명(상호명)을 추출하세요.
6. location은 영수증에 표기된 시/구/동 등 지역명을 추출하세요. 없으면 빈 문자열("")로 출력하세요.
7. category는 아래 기준으로 반드시 5가지 중 1개만 선택하세요:
   - 식대: 식당, 카페, 편의점 식품, 배달 음식 등 식음료 관련
   - 판촉: 거래처·병원 등 외부 대상에게 제공하는 선물, 간식, 판촉물
   - 소모품비: 사무용품, 문구, 청소용품, 전자소모품 등
   - 차량유지비: 주유, 주차, 톨게이트, 세차, 차량 수리 등
   - 기타: 위 카테고리에 해당하지 않는 모든 지출
8. purpose는 결제 목적과 상세 사유를 50자 이내로 작성하세요. (예: 외부 미팅 후 식사, 거래처 직원 간식 제공)
9. attendees는 영수증에서 알 수 없으므로 빈 문자열("")로 출력하세요.
10. 영수증이 아닌 이미지의 경우에도 최대한 정보를 추출하려 시도하세요.

출력 JSON 형식:
{"date": "YYYY-MM-DD", "time": "HH:MM", "merchant": "가맹점명", "location": "지역명", "amount": 0, "category": "카테고리", "purpose": "목적/상세사유", "attendees": ""}
"""

# ── CSV 유틸 ──────────────────────────────────────────────────────────────────
CSV_COLS = ["id", "author", "date", "time", "merchant", "location", "amount", "category", "purpose", "attendees", "registered_at"]

def load_csv() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in CSV_COLS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=CSV_COLS)


def save_to_csv(row: dict):
    df = load_csv()
    row["id"] = int(df["id"].max() + 1) if not df.empty and pd.notna(df["id"].max()) else 1
    row["registered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


# ── Gemini Vision ─────────────────────────────────────────────────────────────
def extract_receipt_info(image_bytes: bytes, mime_type: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return {}

    client = genai.Client(api_key=api_key)
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    text_part = types.Part.from_text(
        text=SYSTEM_PROMPT + "\n\n첨부된 영수증 이미지를 분석하여 위 지침에 따라 JSON만 출력하세요."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[text_part, image_part])],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(raw)


# ── 페이지: 영수증 등록 ──────────────────────────────────────────────────────
def page_register():
    st.title("📄 영수증 등록")

    author = st.selectbox("작성자", AUTHORS, key="author_select")
    uploaded = st.file_uploader("영수증 이미지 업로드", type=["jpg", "jpeg", "png"])

    extracted: dict = {}

    if uploaded:
        image_bytes = uploaded.read()
        mime_type = "image/jpeg" if uploaded.type in ("image/jpg", "image/jpeg") else "image/png"

        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image_bytes, caption="업로드된 영수증", use_container_width=True)

        with col_info:
            with st.spinner("🤖 Gemini AI가 영수증을 분석 중입니다..."):
                try:
                    extracted = extract_receipt_info(image_bytes, mime_type)
                    st.success("✅ 자동 추출 완료! 아래 내용을 확인·수정 후 제출하세요.")
                except Exception as e:
                    st.error(f"AI 분석 실패: {e}")

    st.divider()
    st.subheader("📝 지출 정보 확인 및 수정")

    with st.form("receipt_form"):
        # 날짜 + 시간
        col1, col2 = st.columns(2)
        with col1:
            date_val = extracted.get("date", datetime.today().strftime("%Y-%m-%d"))
            try:
                date_obj = datetime.strptime(date_val, "%Y-%m-%d").date()
            except Exception:
                date_obj = datetime.today().date()
            date_input = st.date_input("사용일자", value=date_obj)

        with col2:
            time_val = extracted.get("time", "")
            try:
                time_obj = datetime.strptime(time_val, "%H:%M").time()
            except Exception:
                time_obj = dtime(0, 0)
            time_input = st.time_input("사용시간", value=time_obj)

        # 가맹점 + 지역
        col3, col4 = st.columns(2)
        with col3:
            merchant_input = st.text_input("가맹점명", value=extracted.get("merchant", ""))
        with col4:
            location_input = st.text_input("결제지역(시/구)", value=extracted.get("location", ""))

        # 금액 + 카테고리
        col5, col6 = st.columns(2)
        with col5:
            amount_input = st.number_input(
                "결제금액(원)", min_value=0, step=100,
                value=int(extracted.get("amount", 0))
            )
        with col6:
            default_cat = extracted.get("category", CATEGORIES[0])
            cat_idx = CATEGORIES.index(default_cat) if default_cat in CATEGORIES else 0
            category_input = st.selectbox("계정과목(분류)", CATEGORIES, index=cat_idx)

        # 목적/사유
        purpose_input = st.text_input(
            "목적/상세사유",
            value=extracted.get("purpose", ""),
            placeholder="예: 외부 미팅 후 식사, 거래처 직원 간식 제공"
        )

        # 참석자/대상처
        attendees_input = st.text_input(
            "참석자/대상처",
            value=extracted.get("attendees", ""),
            placeholder="예: 대표님, 권혁제 / 이지의원"
        )

        submitted = st.form_submit_button("✅ 제출하기", use_container_width=True)

    if submitted:
        if not merchant_input:
            st.warning("가맹점명을 입력해주세요.")
        elif amount_input <= 0:
            st.warning("결제금액을 입력해주세요.")
        else:
            row = {
                "author": author,
                "date": date_input.strftime("%Y-%m-%d"),
                "time": time_input.strftime("%H:%M"),
                "merchant": merchant_input,
                "location": location_input,
                "amount": amount_input,
                "category": category_input,
                "purpose": purpose_input,
                "attendees": attendees_input,
            }
            save_to_csv(row)
            st.success(f"🎉 **{merchant_input}** 영수증이 저장되었습니다!")
            st.balloons()


# ── 페이지: 지출 대시보드 ────────────────────────────────────────────────────
def page_dashboard():
    st.title("📊 지출 대시보드")

    df = load_csv()

    if df.empty:
        st.info("아직 등록된 영수증이 없습니다. '영수증 등록' 메뉴에서 먼저 등록해주세요.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # ── 이번 달 총 지출 ─────────────────────────────────────────────────────
    now = datetime.now()
    this_month = df[(df["date"].dt.year == now.year) & (df["date"].dt.month == now.month)]
    total_this_month = int(this_month["amount"].sum())
    total_all = int(df["amount"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("이번 달 총 지출", f"₩{total_this_month:,}", f"{len(this_month)}건")
    col2.metric("이번 달 식대", f"₩{int(this_month[this_month['category']=='식대']['amount'].sum()):,}")
    col3.metric("이번 달 판촉", f"₩{int(this_month[this_month['category']=='판촉']['amount'].sum()):,}")
    col4.metric("누적 총 지출", f"₩{total_all:,}")

    st.divider()

    # ── 그래프 ──────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("👤 작성자별 지출")
        by_author = df.groupby("author")["amount"].sum().reset_index()
        fig_author = px.pie(
            by_author, values="amount", names="author",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_author.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_author, use_container_width=True)

    with col_b:
        st.subheader("🏷️ 계정과목별 지출")
        by_cat = df.groupby("category")["amount"].sum().reset_index()
        fig_cat = px.bar(
            by_cat, x="category", y="amount",
            color="category",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"amount": "금액 (원)", "category": "계정과목"},
        )
        fig_cat.update_layout(showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── 월별 지출 추이 ───────────────────────────────────────────────────────
    st.subheader("📅 월별 지출 추이")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    by_month = df.groupby("year_month")["amount"].sum().reset_index().sort_values("year_month")
    fig_month = px.line(
        by_month, x="year_month", y="amount", markers=True,
        labels={"amount": "금액 (원)", "year_month": "연월"},
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.divider()

    # ── 전체 데이터 테이블 ───────────────────────────────────────────────────
    st.subheader("📋 전체 영수증 목록")

    display_df = df.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df = display_df.sort_values("id", ascending=False).reset_index(drop=True)
    display_df["amount"] = display_df["amount"].apply(lambda x: f"₩{int(x):,}")

    show_cols = [c for c in ["id", "author", "date", "time", "merchant", "location", "amount", "category", "purpose", "attendees"] if c in display_df.columns]
    col_labels = {
        "id": "No", "author": "작성자", "date": "사용일자", "time": "시간",
        "merchant": "가맹점명", "location": "결제지역", "amount": "결제금액",
        "category": "계정과목", "purpose": "목적/사유", "attendees": "참석자/대상처"
    }
    st.dataframe(
        display_df[show_cols].rename(columns=col_labels),
        use_container_width=True, hide_index=True
    )

    # ── 영수증 삭제 ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🗑️ 영수증 삭제")

    sorted_df = df.sort_values("id", ascending=False)
    delete_options = {
        f"[{int(row['id'])}] {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '-'} | {row.get('merchant','?')} | ₩{int(row['amount']):,}": int(row["id"])
        for _, row in sorted_df.iterrows()
    }

    selected_label = st.selectbox(
        "삭제할 영수증 선택", options=list(delete_options.keys()),
        index=None, placeholder="삭제할 항목을 선택하세요..."
    )

    if selected_label:
        selected_id = delete_options[selected_label]
        target = df[df["id"] == selected_id].iloc[0]
        st.warning(f"**{target.get('merchant','?')}** | {target['date'].strftime('%Y-%m-%d') if pd.notna(target['date']) else '-'} | ₩{int(target['amount']):,} — 이 항목을 삭제하시겠습니까?")

        if st.button("🗑️ 삭제 확인", type="primary", use_container_width=True):
            remaining = df[df["id"] != selected_id]
            remaining.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            st.success("삭제되었습니다.")
            st.rerun()

    # ── CSV 다운로드 ─────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="⬇️ CSV 다운로드",
        data=csv_bytes,
        file_name=f"법인카드사용내역_{now.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="법인카드 영수증 관리",
        page_icon="🧾",
        layout="wide",
    )

    with st.sidebar:
        st.title("🧾 법인카드 관리")
        st.caption("영수증 지출 관리 시스템")
        st.divider()
        menu = st.radio(
            "메뉴",
            ["📄 영수증 등록", "📊 지출 대시보드"],
            label_visibility="collapsed",
        )

    if menu == "📄 영수증 등록":
        page_register()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
