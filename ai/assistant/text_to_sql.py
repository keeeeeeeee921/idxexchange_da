"""
M2: Text-to-SQL assistant (DuckDB + shared LLM client)
======================================================
Natural-language question -> DuckDB SQL (SELECT-only) -> DataFrame, over the
cleaned MLS sold table. Powers the Streamlit chat-with-data app (app.py).

Safety: only a single SELECT/WITH statement is allowed; DDL/DML is rejected and
every query is wrapped in an outer LIMIT so a stray query can't dump 349K rows.

    from ai.assistant.text_to_sql import ask, get_connection
    con = get_connection()
    res = ask("2026-04 各县中位成交价最高的前5个", con)
    print(res["sql"]); print(res["df"])
"""

import os
import re
import sys
import json

import duckdb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)  # so `ai`/`connectors` resolve when run directly

from ai.shared import llm
SOLD_CSV = os.path.join(BASE_DIR, "data", "processed", "tableau", "tableau_sold.csv")
TABLE = "sold"

# Curated column semantics — gives the model the meaning behind each column so it
# writes correct SQL (e.g. yr_mo is text 'YYYY-MM', prices are USD).
COLUMN_NOTES = {
    "CountyOrParish": "县名",
    "City": "城市",
    "PostalCode": "邮编 (ZIP)",
    "MLSAreaMajor": "MLS 区域",
    "PropertyType": "物业类型（均为 Residential）",
    "PropertySubType": "物业细分（如 SingleFamilyResidence, Condominium）",
    "ListOfficeName": "挂牌经纪公司",
    "BuyerOfficeName": "买方经纪公司",
    "ListAgentFullName": "挂牌经纪人",
    "CloseDate": "成交日期",
    "yr_mo": "成交年月，文本，格式 'YYYY-MM'（如 '2026-04'）",
    "close_year": "成交年（整数）",
    "close_month": "成交月（1-12 整数）",
    "ClosePrice": "成交价（美元）",
    "ListPrice": "挂牌价（美元）",
    "OriginalListPrice": "原始挂牌价（美元）",
    "LivingArea": "居住面积（平方英尺）",
    "BedroomsTotal": "卧室数",
    "BathroomsTotalInteger": "卫浴数",
    "DaysOnMarket": "在售天数",
    "price_ratio": "成交价 / 挂牌价",
    "close_to_original_list_ratio": "成交价 / 原始挂牌价",
    "price_per_sqft": "每平方英尺单价（美元）",
    "listing_to_contract_days": "挂牌到签约天数",
    "contract_to_close_days": "签约到成交天数",
    "Latitude": "纬度",
    "Longitude": "经度",
}

SQL_SYSTEM = (
    "You are a senior data analyst who writes correct DuckDB SQL. "
    "Output STRICT JSON only — no prose, no markdown."
)

# DDL/DML and side-effecting keywords are never allowed.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|"
    r"export|import|install|load|call|set|truncate|replace|grant)\b",
    re.IGNORECASE,
)


def get_connection(csv_path=SOLD_CSV):
    """In-memory DuckDB with the sold CSV loaded as table `sold`."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM read_csv_auto('{csv_path}')")
    return con


def schema_prompt(con):
    desc = con.execute(f"DESCRIBE {TABLE}").fetchall()  # (name, type, null, key, default, extra)
    lines = []
    for row in desc:
        col, typ = row[0], row[1]
        note = COLUMN_NOTES.get(col, "")
        lines.append(f"  {col} {typ}" + (f"  -- {note}" if note else ""))
    return f"表名：{TABLE}（加州 CRMLS 住宅成交记录，约 35 万行）\n列：\n" + "\n".join(lines)


def render_sql_prompt(question, schema):
    return (
        f"{schema}\n\n"
        f"用户问题：{question}\n\n"
        "规则：\n"
        f"- 只能查询表 `{TABLE}` 和上面列出的列；不要假设其它表或列。\n"
        "- 只写一条 SELECT 查询（可用 WITH）。禁止 INSERT/UPDATE/DELETE/CREATE 等。\n"
        "- 月份筛选用 yr_mo（文本 'YYYY-MM'，如 yr_mo = '2026-04'）或 close_year/close_month。\n"
        "- 中位数用 median(列)；计数用 count(*)；金额已是美元。\n"
        "- 需要排名/top N 时加 ORDER BY 和 LIMIT。\n\n"
        '只输出 JSON：{"sql": "<一条 DuckDB SELECT 查询>"}'
    )


def _extract_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start:end + 1]
    return json.loads(t)


def is_safe_select(sql):
    """Allow only a single SELECT/WITH statement; reject DDL/DML and multi-statement."""
    s = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^\s*(with|select)\b", s):
        return False
    if ";" in s:                      # no stacked statements
        return False
    if _FORBIDDEN.search(s):
        return False
    return True


def nl_to_sql(question, con, provider=None, model=None):
    raw = llm.complete(
        render_sql_prompt(question, schema_prompt(con)),
        system=SQL_SYSTEM, force_json=True,
        provider=provider, model=model, temperature=0.0,
    )
    return _extract_json(raw)["sql"].strip()


def ask(question, con=None, max_rows=2000, provider=None, model=None):
    """Return {question, sql, df, error}. Never raises — errors come back in
    the dict so the UI can show them."""
    own = con is None
    con = con or get_connection()
    sql = None
    try:
        sql = nl_to_sql(question, con, provider=provider, model=model)
        if not is_safe_select(sql):
            return {"question": question, "sql": sql, "df": None,
                    "error": "生成的 SQL 未通过安全校验（仅允许单条 SELECT）。"}
        guarded = f"SELECT * FROM (\n{sql.rstrip(';')}\n) AS _q LIMIT {max_rows}"
        df = con.execute(guarded).df()
        return {"question": question, "sql": sql, "df": df, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"question": question, "sql": sql, "df": None, "error": str(e)}
    finally:
        if own:
            con.close()


def main():
    con = get_connection()
    for q in [
        "2026-04 各县中位成交价最高的前5个县",
        "2025 年每个月的成交量",
    ]:
        print("\n" + "=" * 70 + f"\nQ: {q}")
        res = ask(q, con)
        if res["error"]:
            print("ERROR:", res["error"])
            print("SQL:", res["sql"])
        else:
            print("SQL:", res["sql"])
            print(res["df"].head(10).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
