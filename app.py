import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
import requests
import json
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="我的第二大腦", layout="wide")

# ===== 基本設定 =====
SHEET_ID = "1szgscKyTQ49LWlG3qFXWOSK8hjWuvek_oyb47rrTW3k"
MISC_SHEET_NAME = "雜記收件匣"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSION = 768

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


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
    with st.container(border=True):
        caption = f"{row.get('記錄時間', '')} ・ {row.get('類別', '') or '未分類'}"
        if show_score is not None:
            caption += f" ・ 相似度 {show_score:.2f}"
        st.caption(caption)

        st.write(row.get("AI摘要") or row.get("內容") or "（尚未整理，批次處理可能還沒跑過這則）")

        img_url = drive_image_url(row.get("照片連結"))
        if img_url:
            st.image(img_url, width=300)

        ai_tags = row.get("AI標籤", "")
        manual_tags = row.get("手動標籤", "")
        all_tags = "、".join([t for t in [ai_tags, manual_tags] if t])
        if all_tags:
            st.caption(f"標籤：{all_tags}")

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
st.title("🧠 我的第二大腦")

with st.sidebar:
    st.subheader("設定")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="搜尋功能需要用到；跟你其他工具用同一把就可以",
    )
    if st.button("🔄 重新整理資料"):
        load_misc_data.clear()
        st.rerun()

df = load_misc_data()

if df.empty:
    st.info("雜記收件匣目前沒有資料，先去 LINE 傳幾則記錄看看吧！")
    st.stop()

tab_search, tab_browse = st.tabs(["🔍 搜尋", "📂 瀏覽"])

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

    st.write(f"共 {len(filtered)} 則")

    for _, row in filtered.iterrows():
        render_entry(row)
