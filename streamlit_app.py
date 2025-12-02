import numpy as np
import pandas as pd
from collections import Counter

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from mundartchat_data import (
    RANDOM_STATE,
    DATA_CSV_BASE,
    DATA_CSV_CHATPAIRS,
    LABEL_ORDER,
    TOKEN_PATTERN,
    preprocess_text_chat,
    build_base_dataset,
    build_chatpairs_dataset,
)

# =========================================================
# Globale Config
# =========================================================

SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32


# =========================================================
# Daten laden / erstellen (cached)
# =========================================================

@st.cache_data
def load_datasets():
    """Basis- und Chatpair-Datensätze laden, bei Bedarf neu erstellen."""
    try:
        base_df = pd.read_csv(DATA_CSV_BASE)
    except FileNotFoundError:
        base_df = build_base_dataset()

    # Sicherstellen, dass text_clean existiert
    if "text_clean" not in base_df.columns:
        base_df["text_clean"] = base_df["text"].astype(str).apply(preprocess_text_chat)

    try:
        resp_df = pd.read_csv(DATA_CSV_CHATPAIRS)
    except FileNotFoundError:
        resp_df = build_chatpairs_dataset()

    return base_df, resp_df


# =========================================================
# Training (cached – wird nur einmal pro Session gemacht)
# =========================================================

def tokenize_for_lm(text: str):
    clean = preprocess_text_chat(text)
    if not clean:
        return []
    return clean.split()


def train_ngram_lm(texts, n_max: int = 3):
    ngram_counts = {n: Counter() for n in range(1, n_max + 1)}
    for t in texts:
        toks = tokenize_for_lm(t)
        if not toks:
            continue
        toks = ["<s>"] + toks + ["</s>"]
        for n in range(1, n_max + 1):
            if len(toks) < n:
                continue
            for i in range(len(toks) - n + 1):
                ngram = tuple(toks[i:i + n])
                ngram_counts[n][ngram] += 1

    def lm_analyzer(text: str):
        return tokenize_for_lm(text)

    return ngram_counts, lm_analyzer


def _is_good_token(tok: str) -> bool:
    # keine Satzgrenzen, keine Placeholder, kein 1-Zeichen-Rauschen
    if tok in ("<s>", "</s>"):
        return False
    if tok.startswith("<") and tok.endswith(">"):
        return False
    if len(tok) < 2:
        return False
    return True


def next_word_candidates(prefix, ngram_counts, analyzer, n_max=3, topk=5):
    toks = analyzer(preprocess_text_chat(prefix))
    backoff_level = None  # NEW

    for n in range(n_max, 0, -1):
        if n == 1:
            # Unigram
            total = sum(cnt for (tok,), cnt in ngram_counts[1].items() if _is_good_token(tok))
            if total == 0:
                continue
            backoff_level = 1  # NEW
            candidates = [
                (tok, cnt)
                for (tok,), cnt in ngram_counts[1].most_common()
                if _is_good_token(tok)
            ][:topk]
            return candidates, total, backoff_level  # NEW

        # 2-gram & 3-gram
        if len(toks) < n - 1:
            continue

        context = tuple(toks[-(n - 1):])
        candidates = []
        for ng, cnt in ngram_counts[n].items():
            if ng[:-1] == context:
                w = ng[-1]
                if _is_good_token(w):
                    candidates.append((w, cnt))

        if candidates:
            backoff_level = n  # NEW
            candidates.sort(key=lambda x: x[1], reverse=True)
            total_cnt = sum(c for _, c in candidates)
            return candidates[:topk], total_cnt, backoff_level  # NEW

    return [], 0, None


@st.cache_resource
def train_all_models(base_df: pd.DataFrame, resp_df: pd.DataFrame):
    """Trainiert Klassifikationsmodelle, LM und Retrieval-Komponenten."""
    # -------- Split für Klassifikation --------
    X_tr_clean, X_te_clean, y_train, y_test = train_test_split(
        base_df["text_clean"],
        base_df["label"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=base_df["label"],
    )

    X_tr_raw = base_df.loc[X_tr_clean.index, "text"]
    X_te_raw = base_df.loc[X_te_clean.index, "text"]

    # -------- BoW / TF-IDF --------
    bow = Pipeline([
        ("vec", CountVectorizer(token_pattern=TOKEN_PATTERN)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]).fit(X_tr_clean, y_train)

    tfidf = Pipeline([
        ("vec", TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]).fit(X_tr_clean, y_train)

    # -------- SBERT + LR --------
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    X_list_tr = pd.Series(X_tr_raw).astype(str).tolist()
    emb_train = sbert_model.encode(
        X_list_tr,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )
    sbert_clf = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    ).fit(emb_train, y_train)

    # -------- Evaluation (nur einmal, in Panel anzeigen) --------
    X_list_te = pd.Series(X_te_raw).astype(str).tolist()
    emb_test = sbert_model.encode(
        X_list_te,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )
    y_pred_sbert = sbert_clf.predict(emb_test)

    eval_info = {}

    # BoW
    y_pred_bow = bow.predict(X_te_clean)
    eval_info["bow"] = {
        "report": classification_report(y_test, y_pred_bow, digits=3),
        "accuracy": accuracy_score(y_test, y_pred_bow),
    }

    # TF-IDF
    y_pred_tfidf = tfidf.predict(X_te_clean)
    eval_info["tfidf"] = {
        "report": classification_report(y_test, y_pred_tfidf, digits=3),
        "accuracy": accuracy_score(y_test, y_pred_tfidf),
    }

    # SBERT
    eval_info["sbert"] = {
        "report": classification_report(y_test, y_pred_sbert, digits=3),
        "accuracy": accuracy_score(y_test, y_pred_sbert),
    }

    # -------- N-Gramm LM --------
    ngram_counts, lm_analyzer = train_ngram_lm(base_df["text_clean"], n_max=3)

    # -------- Antwort-Retrieval --------
    resp_df = resp_df[
        resp_df["answer_mundart"].astype(str).str.len() > 0
    ].reset_index(drop=True)

    resp_emb = sbert_model.encode(
        resp_df["user_text"].astype(str).tolist(),
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )

    models = {
        "bow": bow,
        "tfidf": tfidf,
        "sbert_model": sbert_model,
        "sbert_clf": sbert_clf,
        "ngram_counts": ngram_counts,
        "lm_analyzer": lm_analyzer,
        "resp_df": resp_df,
        "resp_emb": resp_emb,
        "eval_info": eval_info,
    }
    return models


# =========================================================
# Inferenz-Helper
# =========================================================

def probs_pipeline(model, texts):
    vec = model.named_steps["vec"]
    clf = model.named_steps["clf"]
    X = vec.transform(texts)
    P = clf.predict_proba(X)
    cls = clf.classes_
    out = []
    for p in P:
        out.append({c: float(p[i]) for i, c in enumerate(cls)})
    return out


def sbert_predict(models, texts):
    sbert_model = models["sbert_model"]
    sbert_clf = models["sbert_clf"]
    X = pd.Series(texts).astype(str).tolist()
    emb = sbert_model.encode(
        X,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )
    return sbert_clf.predict(emb)


def sbert_predict_proba(models, texts):
    sbert_model = models["sbert_model"]
    sbert_clf = models["sbert_clf"]
    X = pd.Series(texts).astype(str).tolist()
    emb = sbert_model.encode(
        X,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )
    P = sbert_clf.predict_proba(emb)
    cls = sbert_clf.classes_
    out = []
    for p in P:
        out.append({c: float(p[i]) for i, c in enumerate(cls)})
    return out


def format_probs(prob_dict, order=LABEL_ORDER, ndigits=2) -> str:
    return " | ".join(
        f"{lbl}: {prob_dict.get(lbl, 0.0):.{ndigits}f}" for lbl in order
    )


def classify_text(models, raw_inp: str):
    clean_inp = preprocess_text_chat(raw_inp)

    bow = models["bow"]
    tfidf = models["tfidf"]

    bow_pred = bow.predict([clean_inp])[0]
    tfidf_pred = tfidf.predict([clean_inp])[0]
    sbert_pred = sbert_predict(models, [raw_inp])[0]

    bow_probs = probs_pipeline(bow, [clean_inp])[0]
    tfidf_probs = probs_pipeline(tfidf, [clean_inp])[0]
    sbert_probs = sbert_predict_proba(models, [raw_inp])[0]

    return {
        "bow_pred": bow_pred,
        "tfidf_pred": tfidf_pred,
        "sbert_pred": sbert_pred,
        "bow_probs": bow_probs,
        "tfidf_probs": tfidf_probs,
        "sbert_probs": sbert_probs,
    }


def generate_answer(models,
                    user_text: str,
                    predicted_label: str | None = None,
                    topk: int = 5,
                    min_sim: float = 0.2):
    resp_df = models["resp_df"]
    resp_emb = models["resp_emb"]
    sbert_model = models["sbert_model"]

    if len(resp_df) == 0:
        return None, None

    q_emb = sbert_model.encode([user_text], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, resp_emb)[0]

    candidate_idx = np.arange(len(resp_df))
    if predicted_label is not None and "label" in resp_df.columns:
        mask = (resp_df["label"].astype(str) == str(predicted_label))
        if mask.any():
            candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        candidate_idx = np.arange(len(resp_df))

    sims_sub = sims[candidate_idx]
    top_local = np.argsort(-sims_sub)[:min(topk, len(sims_sub))]

    best_local_idx = top_local[0]
    best_idx = candidate_idx[best_local_idx]
    best_sim = float(sims_sub[best_local_idx])
    best_answer = resp_df.iloc[best_idx]["answer_mundart"]

    if best_sim < min_sim:
        return None, best_sim

    return best_answer, best_sim


def debug_neighbors(models,
                    raw_inp: str,
                    topn: int = 5,
                    filter_by_label: bool = True):
    resp_df = models["resp_df"]
    resp_emb = models["resp_emb"]
    sbert_model = models["sbert_model"]

    sbert_label = sbert_predict(models, [raw_inp])[0]

    q_emb = sbert_model.encode([raw_inp], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, resp_emb)[0]

    candidate_idx = np.arange(len(resp_df))
    if filter_by_label and "label" in resp_df.columns:
        mask = (resp_df["label"].astype(str) == str(sbert_label))
        if mask.any():
            candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        return sbert_label, []

    sims_sub = sims[candidate_idx]
    order = np.argsort(-sims_sub)[:min(topn, len(sims_sub))]

    neighbors = []
    for local_idx in order:
        idx = candidate_idx[local_idx]
        row = resp_df.iloc[idx]
        sim = sims_sub[local_idx]

        neighbors.append({
            "similarity": float(sim),
            "user_text": str(row["user_text"]),
            "answer_mundart": str(row["answer_mundart"]),
            "label": row.get("label", "?"),
            "intent": row.get("intent", "?"),
            "is_seed": bool(row.get("is_seed", False)),
        })

    return sbert_label, neighbors


# =========================================================
# Streamlit UI
# =========================================================

def main():
    st.set_page_config(
        page_title="Mundart-Chat Demo",
        page_icon="💬",
        layout="wide",
    )

    st.title("Mundart-Chat Demo")
    st.caption("Sentiment, Next-Word, Antwort-Retrieval für Schweizerdeutsch-Chat")

    with st.sidebar:
        st.header("Setup / Training")

        base_df, resp_df = load_datasets()
        st.write(f"📚 Basisdaten: {len(base_df)} Beispiele")
        st.write(f"💬 Chatpairs: {len(resp_df)} Paare")

        with st.spinner("Modelle werden geladen / trainiert ..."):
            models = train_all_models(base_df, resp_df)

        eval_info = models["eval_info"]
        with st.expander("🔍 Modell-Performance (Testset)", expanded=False):
            for name, info in eval_info.items():
                st.subheader(name.upper())
                st.write(f"Accuracy: **{info['accuracy']:.3f}**")
                st.text(info["report"])

        st.markdown("---")
        st.markdown("**Hinweis:** Modelle werden gecached und nur bei App-Neustart neu trainiert.")

    # ---------------- Eingabe ----------------
    user_text = st.text_area(
        "Mundart-Nachricht eingeben",
        height=120,
        placeholder="z.B. «ich ha kei bock meh uf dä stress»",
    )

    if not user_text.strip():
        st.info("Bitte oben eine Nachricht eingeben.")
        return

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sentiment-Klassifikation",
        "Next-Word Vorschlag",
        "Antwortvorschlag",
        "Debug Nachbarn",
    ])

    # --- Tab 1: Klassifikation ---
    with tab1:
        if st.button("Klassifizieren", key="btn_classify"):
            with st.spinner("Klassifiziere ..."):
                cls = classify_text(models, user_text)

            col1, col2, col3 = st.columns(3)
            col1.metric("BoW", cls["bow_pred"])
            col2.metric("TF-IDF", cls["tfidf_pred"])
            col3.metric("SBERT", cls["sbert_pred"])

            st.subheader("Wahrscheinlichkeiten")
            probs_df = pd.DataFrame([
                {**{"modell": "BoW"}, **cls["bow_probs"]},
                {**{"modell": "TF-IDF"}, **cls["tfidf_probs"]},
                {**{"modell": "SBERT"}, **cls["sbert_probs"]},
            ])
            st.dataframe(probs_df, use_container_width=True)

    # --- Tab 2: Next-Word ---
    with tab2:
        if st.button("Next-Word Vorschläge berechnen", key="btn_nextword"):
            with st.spinner("Berechne Next-Word-Vorschläge ..."):
                cands, backoff = next_word_candidates(   # <-- NEU: 2 Werte
                    user_text,
                    models["ngram_counts"],
                    models["lm_analyzer"],
                    n_max=3,
                    topk=5,
                )
    
            if not cands:
                st.warning("Keine brauchbaren Vorschläge gefunden.")
            else:
                # Info, auf welchem N-Gramm-Level wir gelandet sind
                if backoff == 3:
                    st.info("N-Gramm-Level: 3-Gramm (voller Kontext benutzt)")
                elif backoff == 2:
                    st.info("N-Gramm-Level: 2-Gramm (Backoff – nur letztes Wort)")
                elif backoff == 1:
                    st.info("N-Gramm-Level: 1-Gramm (Unigram-Fallback)")
                else:
                    st.info("N-Gramm-Level: unbekannt / kein Treffer")
    
                rows = []
                for w, p in cands:
                    rows.append({
                        "Token": w,
                        "p (relativ)": round(p, 3),
                        "Vorschlag": (user_text + " " + w).strip(),
                    })
                st.table(pd.DataFrame(rows))

    # --- Tab 3: Antwortvorschlag ---
    with tab3:
        if st.button("Antwort generieren", key="btn_answer"):
            with st.spinner("Klassifiziere & suche passende Antwort ..."):
                cls = classify_text(models, user_text)
                sbert_label = cls["sbert_pred"]
                answer, sim = generate_answer(
                    models,
                    user_text,
                    predicted_label=sbert_label,
                    topk=5,
                    min_sim=0.2,
                )

            st.write(f"**SBERT-Label:** {sbert_label}")
            if answer is None:
                st.warning(
                    f"Keine passende Antwort im Datensatz gefunden "
                    f"(beste Ähnlichkeit: {sim:.2f})."
                )
            else:
                st.subheader("Antwortvorschlag (Mundart)")
                st.success(answer)
                st.caption(f"Ähnlichkeit zu Trainingsbeispielen: {sim:.2f}")

    # --- Tab 4: Debug Nachbarn ---
    with tab4:
        topn = st.slider("Anzahl Nachbarn", min_value=3, max_value=15, value=5)
        if st.button("Ähnlichste Beispiele anzeigen", key="btn_debug"):
            with st.spinner("Suche ähnliche Beispiele ..."):
                sbert_label, neighbors = debug_neighbors(
                    models,
                    user_text,
                    topn=topn,
                    filter_by_label=True,
                )

            st.write(f"**SBERT-Label:** {sbert_label}")
            if not neighbors:
                st.warning("Keine passenden Nachbarn gefunden.")
            else:
                df_neighbors = pd.DataFrame(neighbors)
                # is_seed im UI ausblenden
                df_neighbors = df_neighbors.drop(columns=["is_seed"], errors="ignore")


if __name__ == "__main__":
    main()
