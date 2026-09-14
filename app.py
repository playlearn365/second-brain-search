import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
import requests
import json
import numpy as np
import pandas as pd
import re

st.set_page_config(page_title="我的第二大腦", page_icon="◆", layout="wide")

# ===== 基本設定 =====
SHEET_ID = "1szgscKyTQ49LWlG3qFXWOSK8hjWuvek_oyb47rrTW3k"
MISC_SHEET_NAME = "雜記收件匣"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSION = 768

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ===== 視覺主題：冷色調、俐落 =====
INK = "#E7EAF0"
INK_SOFT = "#7C8698"
PAPER_BG = "#12151B"
CARD_BG = "#1B1F27"
LINE = "#2A2F3B"
ACCENT = "#5E85A6"

CATEGORY_PALETTE = ["#5E85A6", "#6C7A96", "#3F7068", "#6A6EA0", "#5C8B99", "#47607A"]


def category_color(category):
    if not category:
        return INK_SOFT
    idx = sum(ord(c) for c in category) % len(CATEGORY_PALETTE)
    return CATEGORY_PALETTE[idx]


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {INK};
    }}

    .stApp {{
        background-color: {PAPER_BG};
    }}

    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif;
        color: {INK};
        letter-spacing: -0.01em;
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
        border-radius: 2px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
    }}
    .sb-time {{
        color: {INK_SOFT};
        font-size: 0.82rem;
    }}
    .sb-score {{
        color: {INK_SOFT};
        font-size: 0.82rem;
    }}
    .sb-summary {{
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 0.5rem;
        color: {INK};
    }}
    .sb-tags {{
        margin-top: 0.4rem;
    }}
    .sb-tag {{
        display: inline-block;
        border: 1px solid {LINE};
        color: {INK_SOFT};
        padding: 0.1rem 0.55rem;
        border-radius: 2px;
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
        border-radius: 3px !important;
        box-shadow: none !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.02rem;
    }}

    .stButton button {{
        background-color: {ACCENT};
        color: {PAPER_BG};
        border: none;
        border-radius: 2px;
        font-family: 'Space Grotesk', sans-serif;
    }}
    .stButton button:hover {{
        background-color: {INK};
        color: {PAPER_BG};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG} !important;
        border-right: 1px solid {LINE} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {INK} !important;
    }}
    section[data-testid="stSidebar"] input {{
        background-color: {PAPER_BG} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== Google 憑證（Sheets 跟 Drive 共用同一組） =====
@st.cache_resource
def get_credentials():
    creds_dict = dict(st.secrets["gcp_service_account"])
    return Credentials.from_service_account_info(creds_dict, scopes=SCOPES)


@st.cache_resource
def get_gspread_client():
    return gspread.authorize(get_credentials())


def get_drive_access_token():
    creds = get_credentials()
    if not creds.valid:
        creds.refresh(GoogleAuthRequest())
    return creds.token


@st.cache_data(ttl=3600)
def fetch_drive_image_bytes(file_id):
    """直接透過Drive API把照片內容抓下來，比公開連結embed穩定"""
    try:
        token = get_drive_access_token()
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


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


def update_manual_tag(row_numbers, new_tag):
    """同一群組的所有列都寫入同一個手動標籤，維持同步"""
    client = get_gspread_client()
    sh = client.open_by_key(SHEET_ID)
    ws = sh.worksheet(MISC_SHEET_NAME)
    headers = ws.row_values(1)
    col_index = headers.index("手動標籤") + 1
    for row_number in row_numbers:
        ws.update_cell(row_number, col_index, new_tag)


def delete_rows(row_numbers):
    """刪除指定的列，從列號大到小刪，避免刪除過程中列號跑掉"""
    client = get_gspread_client()
    sh = client.open_by_key(SHEET_ID)
    ws = sh.worksheet(MISC_SHEET_NAME)
    for row_number in sorted(row_numbers, reverse=True):
        ws.delete_rows(row_number)
    """把某一列從目前的群組裡拆出來，變成自己獨立一則，
    這樣就不會再跟其他不相關的內容顯示在同一張卡片上。
    同時清空舊的AI摘要等欄位，這樣下次批次處理會針對它自己重新整理，
    不會繼續沿用跟別人混在一起時產生的舊摘要。"""
    client = get_gspread_client()
    sh = client.open_by_key(SHEET_ID)
    ws = sh.worksheet(MISC_SHEET_NAME)
    headers = ws.row_values(1)
    updates = {
        "群組ID": f"solo_{row_number}",
        "AI摘要": "",
        "AI標籤": "",
        "類別": "",
        "向量": "",
    }
    for col_name, value in updates.items():
        col_index = headers.index(col_name) + 1
        ws.update_cell(row_number, col_index, value)


def build_group_entries(df):
    """把同一個群組ID的多列，合併成一張卡片要顯示的內容
    （例如照片跟你事後補的文字說明，本來是分開兩列）"""
    groups = {}
    order = []
    for _, row in df.iterrows():
        gid = row.get("群組ID") or f"_solo_{row['_row_number']}"
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(row)

    entries = []
    for gid in order:
        rows = groups[gid]
        row_numbers = [int(r["_row_number"]) for r in rows]
        times = [r.get("記錄時間", "") for r in rows if r.get("記錄時間")]
        category = next((r.get("類別") for r in rows if r.get("類別")), "")
        summary = next((r.get("AI摘要") for r in rows if r.get("AI摘要")), "")

        contents = []
        for r in rows:
            c = (r.get("內容") or "").strip()
            if c and c not in contents:
                contents.append(c)

        image_ids = []
        for r in rows:
            fid = extract_drive_file_id(r.get("照片連結"))
            if fid and fid not in image_ids:
                image_ids.append(fid)

        ai_tags, manual_tags = [], []
        for r in rows:
            ai_tags += [t for t in (r.get("AI標籤") or "").split("、") if t]
            manual_tags += [t for t in (r.get("手動標籤") or "").split("、") if t]
        ai_tags = list(dict.fromkeys(ai_tags))
        manual_tags = list(dict.fromkeys(manual_tags))

        vector = next((r.get("向量") for r in rows if r.get("向量")), "")

        raw_rows = [
            {
                "row_number": int(r["_row_number"]),
                "content": (r.get("內容") or "").strip(),
                "image_id": extract_drive_file_id(r.get("照片連結")),
                "time": r.get("記錄時間", ""),
            }
            for r in rows
        ]

        entries.append({
            "group_id": gid,
            "row_numbers": row_numbers,
            "raw_rows": raw_rows,
            "time": min(times) if times else "",
            "category": category,
            "summary": summary,
            "content": "\n\n".join(contents),
            "image_ids": image_ids,
            "ai_tags": ai_tags,
            "manual_tags": manual_tags,
            "vector": vector,
        })
    return entries


# ===== 照片網址處理 =====
def extract_drive_file_id(url):
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", str(url) if url else "")
    return m.group(1) if m else None


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


# ===== 顯示一張卡片（同一群組的照片+文字合併顯示） =====
def render_entry(entry, show_score=None):
    row_key = "entry_" + "_".join(str(n) for n in entry["row_numbers"])
    category = entry["category"] or "未分類"
    color = category_color(entry["category"])

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
                <span class="sb-time">{entry['time']} {score_html}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary = entry["summary"] or entry["content"]
        if summary:
            st.markdown(f"<div class='sb-summary'>{summary}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='sb-empty'>尚未整理，批次處理可能還沒跑過這則</div>",
                unsafe_allow_html=True,
            )

        for file_id in entry["image_ids"]:
            image_bytes = fetch_drive_image_bytes(file_id)
            if image_bytes:
                st.image(image_bytes, width=320)
            else:
                st.markdown(
                    f"<div class='sb-empty'>照片載入失敗，"
                    f"<a href='https://drive.google.com/file/d/{file_id}/view' target='_blank' "
                    f"style='color:{ACCENT};'>點此在雲端硬碟開啟</a></div>",
                    unsafe_allow_html=True,
                )

        all_tags = entry["ai_tags"] + entry["manual_tags"]
        if all_tags:
            tags_html = "".join(f"<span class='sb-tag'>{t}</span>" for t in all_tags)
            st.markdown(f"<div class='sb-tags'>{tags_html}</div>", unsafe_allow_html=True)

        manual_tags_str = "、".join(entry["manual_tags"])
        with st.expander("查看完整內容 / 補標籤"):
            st.write(entry["content"] or "（沒有文字內容）")
            new_tag = st.text_input(
                "補充手動標籤（用、分隔多個）",
                value=manual_tags_str,
                key=f"tag_input_{row_key}",
            )
            if st.button("儲存標籤", key=f"save_btn_{row_key}"):
                update_manual_tag(entry["row_numbers"], new_tag)
                st.success("已更新，重新整理後會看到")
                load_misc_data.clear()

            if len(entry["raw_rows"]) > 1:
                st.markdown("---")
                st.caption("這張卡片合併了以下幾則，如果有不相關的，可以把它拆出去或刪除：")
                for raw in entry["raw_rows"]:
                    preview = raw["content"] or ("（照片）" if raw["image_id"] else "（空白）")
                    cols = st.columns([4, 1, 1])
                    with cols[0]:
                        st.write(f"{raw['time']}　{preview[:40]}")
                    with cols[1]:
                        if st.button("拆開", key=f"split_{raw['row_number']}"):
                            split_row_from_group(raw["row_number"])
                            st.success("已拆開，重新整理後會看到")
                            load_misc_data.clear()
                    with cols[2]:
                        if st.button("刪除", key=f"del_row_{raw['row_number']}"):
                            delete_rows([raw["row_number"]])
                            st.success("已刪除，重新整理後會看到")
                            load_misc_data.clear()

            st.markdown("---")
            if st.button("🗑️ 刪除整張卡片", key=f"del_group_{row_key}"):
                delete_rows(entry["row_numbers"])
                st.success("已刪除，重新整理後會看到")
                load_misc_data.clear()


# ===== 主畫面 =====
st.markdown(
    """
    <div class="sb-header"><h1>第二大腦</h1></div>
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

entries = build_group_entries(df)

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
                    for entry in entries:
                        vec_str = entry["vector"]
                        if not vec_str:
                            continue
                        try:
                            vec = json.loads(vec_str)
                        except Exception:
                            continue
                        score = cosine_similarity(query_vec, vec)
                        scored.append((score, entry))

                    scored.sort(key=lambda x: x[0], reverse=True)
                    top = scored[:10]

                    if not top:
                        st.info("目前還沒有已經整理好的資料可以搜尋（批次處理可能還沒跑過）")
                    else:
                        for score, entry in top:
                            render_entry(entry, show_score=score)

with tab_browse:
    categories = ["全部"] + sorted({e["category"] for e in entries if e["category"]})
    selected_cat = st.selectbox("依類別篩選", categories)

    filtered = entries if selected_cat == "全部" else [e for e in entries if e["category"] == selected_cat]
    filtered = sorted(filtered, key=lambda e: e["time"], reverse=True)

    st.caption(f"共 {len(filtered)} 則")

    for entry in filtered:
        render_entry(entry)
