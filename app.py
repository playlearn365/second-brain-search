import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import requests
import json
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="我的第二大腦", page_icon="🗂️", layout="wide")

# ===== 基本設定 =====
SHEET_ID = "1szgscKyTQ49LWlG3qFXWOSK8hjWuvek_oyb47rrTW3k"
MISC_SHEET_NAME = "雜記收件匣"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSION = 768

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ===== 視覺主題：像一疊手寫索引卡片 =====
INK = "#3A3226"
INK_SOFT = "#8A7F6C"
PAPER_BG = "#EDE6D6"
CARD_BG = "#F7F2E7"
LINE = "#D9CFB8"

CATEGORY_PALETTE = ["#5B6E4F", "#8C5B3F", "#4C6478", "#8A6A94", "#A9752F", "#5C7A76"]


def category_color(category):
    if not category:
        return INK_SOFT
    idx = sum(ord(c) for c in category) % len(CATEGORY_PALETTE)
    return CATEGORY_PALETTE[idx]


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Figtree:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Figtree', sans-serif;
        color: {INK};
    }}

    .stApp {{
        background-color: {PAPER_BG};
    }}

    h1, h2, h3 {{
        font-family: 'Fraunces', serif;
        color: {INK};
    }}

    .sb-header {{
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }}
    .sb-header h1 {{
        font-size: 2.1rem;
        font-weight: 600;
        margin: 0;
    }}
    .sb-subtitle {{
        color: {INK_SOFT};
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }}

    .sb-card-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }}
    .sb-chip {{
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 3px;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    .sb-time {{
        color: {INK_SOFT};
        font-size: 0.82rem;
    }}
    .sb-score {{
        color: {INK_SOFT};
        font-size: 0.82rem;
        font-style: italic;
    }}
    .sb-summary {{
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }}
    .sb-tags {{
        margin-top: 0.4rem;
    }}
    .sb-tag {{
        display: inline-block;
        border: 1px solid {LINE};
        color: {INK_SOFT};
        padding: 0.1rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
    }}
    .sb-empty {{
        color: {INK_SOFT};
        font-style: italic;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {LINE} !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-family: 'Fraunces', serif;
        font-size: 1.02rem;
    }}

    .stButton button {{
        background-color: {INK};
        color: {CARD_BG};
        border: none;
        border-radius: 3px;
    }}
    .stButton button:hover {{
        background-color: {CATEGORY_PALETTE[1]};
        color: {CARD_BG};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {LINE};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== Google Sheets 連線 =====
@st.cache_resource
def get_gspread_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


@st.cache_data(ttl=60)
def load_misc_data():
    client = get_gspread_client()
    sh = client.open_by_key(SHEET_ID)
    ws = sh.worksheet(MISC_SHEET_NAME)
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    df["_row_number"] = range(2, 2 + len(df))
    return df


def update_manual_tag(row_number, new_tag):
    client = get_gspread_client()
    sh = client.open_by_key(SHEET_ID)
    ws = sh.worksheet(MISC_SHEET_NAME)
    headers = ws.row_values(1)
    col_index = headers.index("手動標籤") + 1
    ws.update_cell(row_number, col_index, new_tag)


# ===== 照片網址處理 =====
def extract_drive_file_id(url):
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", str(url) if url else "")
    return m.group(1) if m else None


def drive_image_url(url):
    file_id = extract_drive_file_id(url)
    if not file_id:
        return None
    return f"https://drive.google.com/uc?export=view&id={file_id}"


# ===== Gemini 向量化 / 搜尋 =====
def embed_text(text, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": f"models/{GEMINI_EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": EMBED_DIMENSION,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Gemini API 錯誤"))
    return data.get("embedding", {}).get("values", [])


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return -1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


# ===== 顯示單則筆記卡片（搜尋跟瀏覽共用） =====
def render_entry(row, show_score=None):
    row_key = f"entry_{row['_row_number']}"
    category = row.get("類別", "") or "未分類"
    color = category_color(row.get("類別", ""))

    with st.container(border=True, key=row_key):
        score_html = (
            f"<span class='sb-score'>相似度 {show_score:.2f}</span>"
            if show_score is not None
            else ""
        )
        st.markdown(
            f"""
            <div class="sb-card-head">
                <span class="sb-chip" style="background:{color}22; color:{color};">{category}</span>
                <span class="sb-time">{row.get('記錄時間', '')} {score_html}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary = row.get("AI摘要") or row.get("內容")
        if summary:
            st.markdown(f"<div class='sb-summary'>{summary}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='sb-empty'>尚未整理，批次處理可能還沒跑過這則</div>",
                unsafe_allow_html=True,
            )

        img_url = drive_image_url(row.get("照片連結"))
        if img_url:
            st.image(img_url, width=320)

        ai_tags = row.get("AI標籤", "")
        manual_tags = row.get("手動標籤", "")
        all_tags = [t for t in (ai_tags.split("、") + manual_tags.split("、")) if t]
        if all_tags:
            tags_html = "".join(f"<span class='sb-tag'>{t}</span>" for t in all_tags)
            st.markdown(f"<div class='sb-tags'>{tags_html}</div>", unsafe_allow_html=True)

        with st.expander("查看完整內容 / 補標籤"):
            st.write(row.get("內容", "") or "（沒有文字內容）")
            new_tag = st.text_input(
                "補充手動標籤（用、分隔多個）",
                value=manual_tags,
                key=f"tag_input_{row['_row_number']}",
            )
            if st.button("儲存標籤", key=f"save_btn_{row['_row_number']}"):
                update_manual_tag(int(row["_row_number"]), new_tag)
                st.success("已更新，重新整理後會看到")
                load_misc_data.clear()


# ===== 主畫面 =====
st.markdown(
    """
    <div class="sb-header"><h1>🗂️ 第二大腦</h1></div>
    <div class="sb-subtitle">你隨手記下的一切，整理成找得到的樣子</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("設定")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="搜尋功能需要用到；跟你其他工具用同一把就可以",
    )
    if st.button("重新整理資料"):
        load_misc_data.clear()
        st.rerun()

df = load_misc_data()

if df.empty:
    st.info("雜記收件匣目前沒有資料，先去 LINE 傳幾則記錄看看吧！")
    st.stop()

tab_search, tab_browse = st.tabs(["搜尋", "瀏覽"])

with tab_search:
    query = st.text_input("想找什麼？直接打字問，比如「之前有沒有看過關於植物的觀察」")

    if query:
        if not api_key:
            st.warning("請先在左側輸入你的 Gemini API Key 才能搜尋")
        else:
            with st.spinner("搜尋中..."):
                try:
                    query_vec = embed_text(query, api_key)
                except Exception as e:
                    st.error(f"搜尋失敗：{e}")
                    query_vec = None

                if query_vec:
                    scored = []
                    for _, row in df.iterrows():
                        vec_str = row.get("向量", "")
                        if not vec_str:
                            continue
                        try:
                            vec = json.loads(vec_str)
                        except Exception:
                            continue
                        score = cosine_similarity(query_vec, vec)
                        scored.append((score, row))

                    scored.sort(key=lambda x: x[0], reverse=True)
                    top = scored[:10]

                    if not top:
                        st.info("目前還沒有已經整理好的資料可以搜尋（批次處理可能還沒跑過）")
                    else:
                        for score, row in top:
                            render_entry(row, show_score=score)

with tab_browse:
    categories = ["全部"] + sorted([c for c in df["類別"].dropna().unique() if c])
    selected_cat = st.selectbox("依類別篩選", categories)

    filtered = df if selected_cat == "全部" else df[df["類別"] == selected_cat]
    filtered = filtered.sort_values("記錄時間", ascending=False)

    st.caption(f"共 {len(filtered)} 則")

    for _, row in filtered.iterrows():
        render_entry(row)
