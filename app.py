"""
Chat-with-Data — CRMLS market assistant (M2)
============================================
A Streamlit app that lets non-technical stakeholders query the MLS sold data in
plain language. Question -> LLM writes DuckDB SQL -> result table + auto chart.

Run:
    .venv/bin/streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai.assistant.text_to_sql import ask, get_connection
from ai.shared import llm

st.set_page_config(page_title="CRMLS Chat-with-Data", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="加载 35 万行成交数据到 DuckDB…")
def _connection():
    return get_connection()


SAMPLES = [
    "2026-04 各县中位成交价最高的前 5 个县",
    "2025 年每个月的成交量",
    "Orange County 各城市的中位单价（price_per_sqft）前 10",
    "2026 年 SingleFamilyResidence 的平均在售天数按县排序",
]


def _auto_chart(df):
    """Best-effort chart for a 2-column (label, number) result."""
    if df is None or df.shape[1] != 2 or len(df) > 60:
        return
    label, value = df.columns[0], df.columns[1]
    if not pd.api.types.is_numeric_dtype(df[value]):
        return
    series = df.set_index(label)[value]
    # time-like first column -> line, else bar
    if df[label].astype(str).str.match(r"^\d{4}(-\d{2})?$").all():
        st.line_chart(series)
    else:
        st.bar_chart(series)


# --- Sidebar ---------------------------------------------------------------
provider = llm.resolve_provider()
model = llm.resolve_model(provider)
with st.sidebar:
    st.header("Chat-with-Data")
    st.caption("用自然语言查询 CRMLS 加州住宅成交数据。")
    st.markdown(f"**LLM：** `{provider}:{model}`")
    st.markdown("**数据表：** `sold`（~35 万行，2024-01 → 2026-04）")
    st.divider()
    st.markdown("**示例问题（点一下填入）**")
    for s in SAMPLES:
        if st.button(s, use_container_width=True):
            st.session_state["q"] = s
    st.divider()
    st.caption("⚠️ 机密 MLS 数据，仅供内部使用。数据与 LLM 调用均在本机完成。")

# --- Main ------------------------------------------------------------------
st.title("🏠 CRMLS 市场数据助手")
st.caption("问一个市场问题 → AI 生成 SQL → 返回结果表与图表。只读，仅允许 SELECT 查询。")

question = st.text_input(
    "你的问题", key="q",
    placeholder="例如：2026-04 各县中位成交价最高的前 5 个县",
)

if question:
    con = _connection()
    with st.spinner(f"{provider}:{model} 正在生成 SQL 并查询…"):
        res = ask(question, con=con, provider=provider, model=model)

    if res["sql"]:
        st.markdown("**生成的 SQL**")
        st.code(res["sql"], language="sql")

    if res["error"]:
        st.error(f"查询失败：{res['error']}")
    else:
        df = res["df"]
        st.markdown(f"**结果**（{len(df):,} 行）")
        st.dataframe(df, use_container_width=True, hide_index=True)
        _auto_chart(df)
        st.download_button(
            "下载 CSV", df.to_csv(index=False).encode("utf-8-sig"),
            file_name="query_result.csv", mime="text/csv",
        )
else:
    st.info("在上方输入问题，或从左侧点一个示例问题开始。")
