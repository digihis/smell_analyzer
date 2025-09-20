import io
import os
import re
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="匂い表現 KWIC ビューア（サブカテゴリ・かな合算・リンクのみ）", layout="wide")

# =========================
# データ読み込み
# =========================
@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def read_csv_auto(file_or_buf) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp932", "shift_jis", "euc-jp"]:
        try:
            df = pd.read_csv(file_or_buf, encoding=enc, low_memory=False)
            df.attrs["encoding_used"] = enc
            return df
        except Exception:
            if hasattr(file_or_buf, "seek"):
                try:
                    file_or_buf.seek(0)
                except Exception:
                    pass
    # fallback: chardet
    try:
        import chardet  # type: ignore
        if hasattr(file_or_buf, "read"):
            b = file_or_buf.read()
            if hasattr(file_or_buf, "seek"):
                try:
                    file_or_buf.seek(0)
                except Exception:
                    pass
        else:
            with open(file_or_buf, "rb") as f:
                b = f.read()
        enc = chardet.detect(b).get("encoding") or "utf-8"
        df = pd.read_csv(io.BytesIO(b), encoding=enc, low_memory=False)
        df.attrs["encoding_used"] = enc + " (chardet)"
        return df
    except Exception:
        st.error("CSVの読み込みに失敗しました。")
        raise

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in ["dic","pos","lemma","url","work_id","page_id","left","right","term","sentence","xywh"]:
        if col not in df.columns:
            df[col] = ""
    if "kwic" not in df.columns:
        df["kwic"] = df["left"].astype(str) + "「" + df["term"].astype(str) + "」" + df["right"].astype(str)
    return df

# =========================
# サブカテゴリ定義
# =========================
def load_smell_categories() -> Dict[str, List[str]]:
    candidates = [
        "smell_words.txt",
        os.path.join(os.getcwd(), "smell_words.txt"),
        "/mnt/data/smell_words.txt",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    cats: Dict[str, List[str]] = {
        "植物・香木": [],
        "風": [],
        "香に関する名詞": [],
        "匂い関連の動詞・形容詞": [],
    }
    if not path:
        return cats
    current = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                name = s.lstrip("#").strip()
                if "植物・香木" in name:
                    current = "植物・香木"
                elif "風" in name:
                    current = "風"
                elif "名詞" in name:
                    current = "香に関する名詞"
                elif ("動詞" in name) or ("形容詞" in name):
                    current = "匂い関連の動詞・形容詞"
                else:
                    current = None
                continue
            if current and s.startswith("^"):
                cats[current].append(s)
    return cats

_SMELL_CATS = load_smell_categories()

def terms_mask_by_category(series: pd.Series, category_name: str) -> pd.Series:
    """term列に対して、該当サブカテゴリ（正規表現群）に合致するかのブールSeriesを返す"""
    pats = _SMELL_CATS.get(category_name, [])
    s = series.astype(str)
    if not pats:
        return pd.Series(False, index=s.index, dtype=bool)
    # 高速化：ORでまとめて1回検索
    try:
        combined = "(?:" + "|".join(pats) + ")"
        return s.str.contains(combined, regex=True, na=False)
    except re.error:
        safe = "(?:" + "|".join(re.escape(p) for p in pats) + ")"
        return s.str.contains(safe, regex=True, na=False)

# =========================
# かな←→漢字 合算設定
# =========================
KANA_TO_KANJI = {
    "うめ": "梅",
    "さくら": "桜",
    "たちばな": "橘",
    "ふじ": "藤",
    "きく": "菊",
    "なでしこ": "撫子",
    "はぎ": "萩",
    "かきつばた": "杜若",
    "びゃくだん": "白檀",
    "じんこう": "沈香",
    "ゆうかぜ": "夕風",
    "にほふ": "匂ふ",
    "にほひ": "匂ひ",
    "かんばしい": "芳しい",
}
KANJI_SET = set(KANA_TO_KANJI.values())

def aggregate_kana_kanji_counts(counts: pd.Series) -> pd.DataFrame:
    agg = {}
    for term, c in counts.items():
        if term in KANA_TO_KANJI:
            canon = KANA_TO_KANJI[term]
            agg.setdefault(canon, {"漢字": 0, "かな": 0})
            agg[canon]["かな"] += int(c)
        elif term in KANJI_SET:
            canon = term
            agg.setdefault(canon, {"漢字": 0, "かな": 0})
            agg[canon]["漢字"] += int(c)
        else:
            canon = term
            agg.setdefault(canon, {"漢字": 0, "かな": 0})
            agg[canon]["漢字"] += int(c)
    rows = []
    total_count = 0
    for canon, parts in agg.items():
        total = parts["漢字"] + parts["かな"]
        total_count += total
        rows.append({"canonical": canon, "segment": "漢字", "count": parts["漢字"], "total": total})
        if parts["かな"]:
            rows.append({"canonical": canon, "segment": "かな", "count": parts["かな"], "total": total})
    df_long = pd.DataFrame(rows)
    if not df_long.empty:
        df_long["canonical"] = pd.Categorical(
            df_long["canonical"],
            categories=(df_long.groupby("canonical")["total"].max().sort_values(ascending=False).index),
            ordered=True,
        )
        df_long.attrs["total_count"] = total_count
    return df_long

# =========================
# サイドバー
# =========================
with st.sidebar:
    st.header("1) データ入力")
    uploaded = st.file_uploader("CSVファイルを選択", type=["csv"])
    st.caption("読み込み時にエンコーディングを自動判定します。")

    st.header("2) 絞り込み（KWICベース）")
    # ※ 説明文は非表示（内部ではマーカー除去で検索）
    term_query = st.text_input("テキスト検索（正規表現）", value="")

st.title("匂い表現 KWIC ビューア")

# =========================
# メイン処理
# =========================
df: Optional[pd.DataFrame] = None
if uploaded:
    df = read_csv_auto(uploaded)
    df = ensure_columns(df)

if df is not None:
    # KWIC検索用（内部ではマーカー「」「」を除去して検索）※UIには説明文を出さない
    s_kwic_all = (df["left"].astype(str) + df["term"].astype(str) + df["right"].astype(str)).str.replace("「|」", "", regex=True)

    # まず「検索のみ」を適用（サマリーやグラフのベース）
    mask_search_only = pd.Series(True, index=df.index, dtype=bool)
    if term_query:
        try:
            mask_search_only &= s_kwic_all.str.contains(term_query, regex=True, na=False)
        except re.error:
            st.error("テキスト検索の正規表現が不正です。")
            mask_search_only &= False
    df_after_search = df[mask_search_only].copy()

    # ===== サマリー =====
    st.subheader("サマリー")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総ヒット数", f"{len(df):,}")
    c2.metric("検索後（KWIC）", f"{len(df_after_search):,}")
    c3.metric("ユニーク文数", f"{df_after_search['sentence'].astype(str).nunique():,}")
    c4.metric("ユニーク lemma 数", f"{df_after_search['lemma'].astype(str).replace('', pd.NA).dropna().nunique():,}")

    # ===== ヒットした単語件数 =====
    st.subheader("ヒットした単語件数")
    charts = st.tabs(["term 別", "lemma 別", "work_id 別（上位）"])

    # ---------- term 別 ----------
    with charts[0]:
        cL, cR = st.columns([3, 1], vertical_alignment="top")
        with cR:
            # ★ サブカテゴリ選択（元の場所＝グラフ右）
            subcat = st.selectbox(
                "サブカテゴリ",
                ["（未選択：すべて）", "植物・香木", "風", "香に関する名詞", "匂い関連の動詞・形容詞"],
                index=0,
            )
            top_n_term = st.number_input("上位表示数", min_value=5, max_value=100, value=30, step=5, key="term_topn")
            st.session_state["subcat_selected"] = subcat

        with cL:
            df_base_for_chart = df_after_search.copy()
            if subcat != "（未選択：すべて）":
                mask_cat_chart = terms_mask_by_category(df_base_for_chart["term"], subcat)
                df_base_for_chart = df_base_for_chart[mask_cat_chart].copy()

            if df_base_for_chart.empty:
                st.info("該当する term がありません。")
            else:
                counts = df_base_for_chart["term"].astype(str).replace("", pd.NA).dropna().value_counts()

                if subcat == "（未選択：すべて）":
                    top_terms = counts.head(int(top_n_term)).reset_index()
                    top_terms.columns = ["term", "count"]
                    st.info(f"合算後の全体件数: {int(counts.sum()):,}")

                    # 並び順（上から多い順）を明示
                    order_terms = top_terms.sort_values("count", ascending=False)["term"].tolist()

                    # 棒 + 右横ラベル
                    bars = alt.Chart(top_terms).mark_bar().encode(
                        x=alt.X("count:Q", title="件数"),
                        y=alt.Y("term:N", title="term", sort=order_terms),
                    )
                    labels = alt.Chart(top_terms).mark_text(
                        align="left", baseline="middle", dx=3
                    ).encode(
                        x="count:Q",
                        y=alt.Y("term:N", sort=order_terms),
                        text=alt.Text("count:Q", format=","),
                    )
                    st.altair_chart(bars + labels, use_container_width=True)

                else:
                    long_df = aggregate_kana_kanji_counts(counts)
                    if long_df.empty:
                        st.info("該当する term がありません。")
                    else:
                        total_count = long_df.attrs.get("total_count", int(long_df["count"].sum()))
                        st.info(f"合算後の全体件数: {total_count:,}")
                        long_df = long_df.sort_values(["total", "segment"], ascending=[False, True])
                        keep = (
                            long_df.groupby("canonical")["total"]
                            .max()
                            .sort_values(ascending=False)
                            .head(int(top_n_term))
                            .index
                        )
                        long_df = long_df[long_df["canonical"].isin(keep)]

                        # 合計で降順の並び順を明示
                        totals = (
                            long_df.groupby("canonical", as_index=False)["count"]
                            .sum()
                            .sort_values("count", ascending=False)
                        )
                        order_canon = totals["canonical"].tolist()

                        # 積み上げ棒
                        bars = alt.Chart(long_df).mark_bar().encode(
                            x=alt.X("sum(count):Q", title="件数"),
                            y=alt.Y("canonical:N", title="term（合算後ラベル）", sort=order_canon),
                            color=alt.Color(
                                "segment:N",
                                title="内訳",
                                scale=alt.Scale(domain=["かな", "漢字"], range=["#F59E0B", "#6366F1"]),
                            ),
                            order=alt.Order("segment", sort="ascending"),
                        )

                        # 合計値ラベル（バー右）
                        labels = (
                            alt.Chart(totals)
                            .mark_text(align="left", baseline="middle", dx=3)
                            .encode(
                                x="count:Q",
                                y=alt.Y("canonical:N", sort=order_canon),
                                text=alt.Text("count:Q", format=","),
                            )
                        )
                        st.altair_chart(bars + labels, use_container_width=True)

    # ---------- lemma 別 ----------
    with charts[1]:
        cL, cR = st.columns([3, 1], vertical_alignment="top")
        with cR:
            top_n_lemma = st.number_input("上位表示数", min_value=5, max_value=100, value=30, step=5, key="lemma_topn")
        with cL:
            top_lemmas = (
                df_after_search["lemma"].astype(str).replace("", pd.NA).dropna()
                .value_counts().head(int(top_n_lemma)).reset_index()
            )
            top_lemmas.columns = ["lemma", "count"]
            if not top_lemmas.empty:
                st.info(f"合算後の全体件数: {top_lemmas['count'].sum():,}")

                order_lemmas = top_lemmas.sort_values("count", ascending=False)["lemma"].tolist()

                bars = alt.Chart(top_lemmas).mark_bar().encode(
                    x=alt.X("count:Q", title="件数"),
                    y=alt.Y("lemma:N", title="lemma", sort=order_lemmas),
                )
                labels = alt.Chart(top_lemmas).mark_text(
                    align="left", baseline="middle", dx=3
                ).encode(
                    x="count:Q",
                    y=alt.Y("lemma:N", sort=order_lemmas),
                    text=alt.Text("count:Q", format=","),
                )
                st.altair_chart(bars + labels, use_container_width=True)
            else:
                st.info("該当する lemma がありません。")

    # ---------- work_id 別 ----------
    with charts[2]:
        cL, cR = st.columns([3, 1], vertical_alignment="top")
        with cR:
            top_n_work = st.number_input("上位表示数", min_value=5, max_value=100, value=30, step=5, key="work_topn")
        with cL:
            top_work = (
                df_after_search["work_id"].astype(str).replace("", pd.NA).dropna()
                .value_counts().head(int(top_n_work)).reset_index()
            )
            top_work.columns = ["work_id", "count"]
            if not top_work.empty:
                st.info(f"合算後の全体件数: {top_work['count'].sum():,}")

                order_work = top_work.sort_values("count", ascending=False)["work_id"].tolist()

                bars = alt.Chart(top_work).mark_bar().encode(
                    x=alt.X("count:Q", title="件数"),
                    y=alt.Y("work_id:N", title="work_id", sort=order_work),
                )
                labels = alt.Chart(top_work).mark_text(
                    align="left", baseline="middle", dx=3
                ).encode(
                    x="count:Q",
                    y=alt.Y("work_id:N", sort=order_work),
                    text=alt.Text("count:Q", format=","),
                )
                st.altair_chart(bars + labels, use_container_width=True)
            else:
                st.info("該当する work_id がありません。")

    # =========================
    # KWIC 表（サブカテゴリ → テキスト検索の順で適用）
    # =========================
    st.subheader("KWIC")
    subcat_current = st.session_state.get("subcat_selected", "（未選択：すべて）")

    show_pos     = st.checkbox("品詞（pos）列を表示", value=True)
    show_lemma   = st.checkbox("基本形（lemma）列を表示", value=True)
    show_page_id = st.checkbox("page_id 列を表示", value=False)

    # 1) サブカテゴリで絞り込み（term に対するフィルタ）
    if subcat_current != "（未選択：すべて）":
        mask_subcat_only = terms_mask_by_category(df["term"], subcat_current)
        df_after_subcat = df[mask_subcat_only].copy()
    else:
        df_after_subcat = df.copy()

    # 2) サブカテゴリ内でKWICテキスト検索（内部的にマーカー除去）
    s_kwic_sub = (
        df_after_subcat["left"].astype(str)
        + df_after_subcat["term"].astype(str)
        + df_after_subcat["right"].astype(str)
    ).str.replace("「|」", "", regex=True)

    mask_text_in_subcat = pd.Series(True, index=df_after_subcat.index, dtype=bool)
    if term_query:
        try:
            mask_text_in_subcat &= s_kwic_sub.str.contains(term_query, regex=True, na=False)
        except re.error:
            st.error("テキスト検索の正規表現が不正です。")
            mask_text_in_subcat &= False
    df_kwic = df_after_subcat[mask_text_in_subcat].copy()

    # 件数（検索前/サブカテゴリ後/サブカテゴリ+検索後）を表示
    n_all = len(df)
    n_after_subcat = len(df_after_subcat)
    n_after_text_in_sub = len(df_kwic)
    st.caption(
        f"件数: 全体 {n_all:,} 件 / サブカテゴリ "
        f"{'全体' if subcat_current=='（未選択：すべて）' else subcat_current} → {n_after_subcat:,} 件"
        + (f" / テキスト検索後 → {n_after_text_in_sub:,} 件" if term_query else "")
    )

    # 表示用KWIC（軽量：素のテキスト）
    def kwic_plain(row):
        return f"{str(row.get('left',''))}「{str(row.get('term',''))}」{str(row.get('right',''))}"

    df_show = df_kwic.copy()
    df_show["KWIC"] = df_show.apply(kwic_plain, axis=1)

    # URLリンク
    def make_url_link(url_val):
        url_str = str(url_val).strip()
        return url_str if url_str and url_str.lower() != "nan" else ""

    df_show["url_link"] = df_show["url"].apply(make_url_link)

    cols = ["KWIC", "term", "url_link", "work_id"]
    if show_page_id: cols.append("page_id")
    if show_pos:     cols.append("pos")
    if show_lemma:   cols.append("lemma")

    try:
        st.dataframe(
            df_show[cols],
            use_container_width=True,
            height=420,
            column_config={
                "url_link": st.column_config.LinkColumn(
                    "国書データベース", help="CSVのurl列のリンク", display_text="該当画像へのリンク"
                )
            },
        )
    except Exception:
        st.dataframe(df_show[cols], use_container_width=True, height=420)

    # ダウンロード（KWICテーブルと同じ内容）
    csv_bytes = df_kwic.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "↓ KWIC表示と同じ内容をCSVでダウンロード",
        data=csv_bytes,
        file_name="smell_kwic_filtered_by_subcat_and_text.csv",
        mime="text/csv",
    )

else:
    st.info("左のサイドバーからCSVを選択してください。")
