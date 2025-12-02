"""
mundart_chat_app.py

Nutzt die Datensätze aus mundart_data.py und bietet:

- Training:
  - BoW + LogisticRegression
  - TF-IDF + LogisticRegression
  - SBERT + LogisticRegression
- 3-Gramm-Language-Model für Next-Word-Vorschläge
- SBERT-Retrieval für Antwortvorschläge
- Interaktive CLI
"""

import numpy as np
import pandas as pd
from collections import Counter

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

# Modell-Config
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 32


# =========================================================
# 1) Training
# =========================================================

def train_and_run():
    print(f"\nLade Basisdaten aus {DATA_CSV_BASE} ...")
    df = pd.read_csv(DATA_CSV_BASE)

    required_base_cols = {"text", "label", "intent", "text_clean"}
    missing_base = required_base_cols - set(df.columns)
    if missing_base:
        raise ValueError(
            f"Fehlende Spalten in {DATA_CSV_BASE}: {missing_base} "
            f"– bitte zuerst Datensätze mit mundart_data.py bauen."
        )

    # zusätzliche Sicherheit: Preprocessing nachziehen
    df["text_clean"] = df["text"].astype(str).apply(preprocess_text_chat)

    print("Basisdaten geladen.")
    print(df.head())
    print("\nLabel-Verteilung:")
    print(df["label"].value_counts())
    print("\nIntent-Verteilung:")
    print(df["intent"].value_counts())

    X_tr_clean, X_te_clean, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    X_tr_raw = df.loc[X_tr_clean.index, "text"]
    X_te_raw = df.loc[X_te_clean.index, "text"]

    # ---------- Klassifikationsmodelle ----------

    def train_bow(X_train, y_train):
        pipe = Pipeline([
            ("vec", CountVectorizer(token_pattern=TOKEN_PATTERN)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
        return pipe.fit(X_train, y_train)

    def train_tfidf(X_train, y_train):
        pipe = Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), token_pattern=TOKEN_PATTERN)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
        return pipe.fit(X_train, y_train)

    def train_sbert(X_train_raw, y_train,
                    model_name=SBERT_MODEL_NAME,
                    batch_size=BATCH_SIZE):
        sbert_model = SentenceTransformer(model_name)
        X_list = pd.Series(X_train_raw).astype(str).tolist()
        emb_train = sbert_model.encode(
            X_list,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
        sbert_clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ).fit(emb_train, y_train)
        return sbert_model, sbert_clf

    print("\nTrainiere BoW-Modell ...")
    bow = train_bow(X_tr_clean, y_train)

    print("Trainiere TF-IDF-Modell ...")
    tfidf = train_tfidf(X_tr_clean, y_train)

    print("Lade / trainiere SBERT + LogisticRegression ...")
    sbert_model, sbert_clf = train_sbert(X_tr_raw, y_train)

    def eval_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    def eval_sbert_model(sbert_model, sbert_clf, X_test_raw, y_test,
                         batch_size=BATCH_SIZE):
        X_list = pd.Series(X_test_raw).astype(str).tolist()
        emb_test = sbert_model.encode(
            X_list,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
        y_pred = sbert_clf.predict(emb_test)
        print("\n=== SBERT-Embeddings + LogisticRegression ===")
        print(classification_report(y_test, y_pred, digits=3))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    eval_model("BoW + LogisticRegression", bow, X_te_clean, y_test)
    eval_model("TF-IDF + LogisticRegression", tfidf, X_te_clean, y_test)
    eval_sbert_model(sbert_model, sbert_clf, X_te_raw, y_test)

    # =====================================================
    # 2) N-Gramm Language Model für Next-Word
    # =====================================================

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

    def next_word_candidates(prefix: str,
                             ngram_counts,
                             analyzer,
                             n_max: int = 3,
                             topk: int = 5):
        toks = analyzer(preprocess_text_chat(prefix))
        # Backoff von n_max -> 1
        for n in range(n_max, 0, -1):
            if n == 1:
                total = sum(
                    cnt
                    for (tok,), cnt in ngram_counts[1].items()
                    if _is_good_token(tok)
                )
                if total == 0:
                    continue
                candidates = [
                    (tok, cnt)
                    for (tok,), cnt in ngram_counts[1].most_common()
                    if _is_good_token(tok)
                ][:topk]
                return [(w, c / float(total)) for w, c in candidates]

            if len(toks) < n - 1:
                continue

            context = tuple(toks[-(n - 1):])
            candidates = []
            for ng, cnt in ngram_counts[n].items():
                if ng[:-1] == context:
                    w = ng[-1]
                    if _is_good_token(w):
                        candidates.append((w, cnt))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:topk]
            total_cnt = sum(c for _, c in candidates)
            return [(w, c / float(total_cnt)) for w, c in candidates]

        return []

    print("\nTrainiere 3-Gramm Language Model für Next-Word-Prediction ...")
    ngram_counts, lm_analyzer = train_ngram_lm(df["text_clean"], n_max=3)
    print("Language Model fertig.")

    # =====================================================
    # 3) Antwort-Retrieval (SBERT)
    # =====================================================

    print(f"\nLade Chatpair-Daten aus {DATA_CSV_CHATPAIRS} ...")
    resp_df = pd.read_csv(DATA_CSV_CHATPAIRS)

    required_resp_cols = {"user_text", "answer_mundart", "label"}
    missing_resp = required_resp_cols - set(resp_df.columns)
    if missing_resp:
        raise ValueError(
            f"Fehlende Spalten in {DATA_CSV_CHATPAIRS}: {missing_resp} "
            f"– bitte zuerst Chatpairs-Datensatz bauen."
        )

    resp_df = resp_df[
        resp_df["answer_mundart"].astype(str).str.len() > 0
    ].reset_index(drop=True)
    print(f"Antwort-Datensatz geladen, Anzahl Paare: {len(resp_df)}")

    print("Berechne SBERT-Embeddings für Antwortkandidaten ...")
    resp_emb = sbert_model.encode(
        resp_df["user_text"].astype(str).tolist(),
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
    )
    print("Embeddings fertig.")

    # =====================================================
    # 4) Helper-Funktionen für Inferenz
    # =====================================================

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

    def sbert_predict(texts, batch_size=BATCH_SIZE):
        X = pd.Series(texts).astype(str).tolist()
        emb = sbert_model.encode(
            X,
            convert_to_numpy=True,
            batch_size=batch_size,
        )
        return sbert_clf.predict(emb)

    def sbert_predict_proba(texts, batch_size=BATCH_SIZE):
        X = pd.Series(texts).astype(str).tolist()
        emb = sbert_model.encode(
            X,
            convert_to_numpy=True,
            batch_size=batch_size,
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

    def generate_answer(user_text: str,
                        predicted_label: str | None = None,
                        topk: int = 5,
                        min_sim: float = 0.2):
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

    # =====================================================
    # 5) Aktionen für CLI
    # =====================================================

    def run_classification(raw_inp: str):
        clean_inp = preprocess_text_chat(raw_inp)

        bow_pred = bow.predict([clean_inp])[0]
        tfidf_pred = tfidf.predict([clean_inp])[0]
        sbert_pred = sbert_predict([raw_inp])[0]

        bow_probs = probs_pipeline(bow, [clean_inp])[0]
        tfidf_probs = probs_pipeline(tfidf, [clean_inp])[0]
        sbert_probs = sbert_predict_proba([raw_inp])[0]

        print("\n— Ergebnisse (Klassifikation) —")
        print("BoW   ->", bow_pred,   " | ", format_probs(bow_probs))
        print("TF-IDF->", tfidf_pred, " | ", format_probs(tfidf_probs))
        print("SBERT ->", sbert_pred, " | ", format_probs(sbert_probs))

        return {
            "bow_pred": bow_pred,
            "tfidf_pred": tfidf_pred,
            "sbert_pred": sbert_pred,
            "bow_probs": bow_probs,
            "tfidf_probs": tfidf_probs,
            "sbert_probs": sbert_probs,
        }

    def run_nextword(raw_inp: str):
        print("\n— Next-Word Vorschläge —")
        cands = next_word_candidates(
            raw_inp,
            ngram_counts,
            lm_analyzer,
            n_max=3,
            topk=5,
        )
        if not cands:
            print("  (keine brauchbaren Vorschläge gefunden)")
        else:
            for w, p in cands:
                suggestion = (raw_inp + " " + w).strip()
                print(f"  {w:15s}  (p ≈ {p:.2f})   →  {suggestion}")

    def run_answer(raw_inp: str, predicted_label: str | None = None):
        answer, sim = generate_answer(
            raw_inp,
            predicted_label=predicted_label,
            topk=5,
            min_sim=0.2,
        )
        print("\n— Antwortvorschlag (Mundart) —")
        if answer is None:
            print("  (keine passende Antwort im Datensatz gefunden)")
            if sim is not None:
                print(f"  [beste Ähnlichkeit lag nur bei: {sim:.2f}]")
        else:
            print(f"  {answer}")
            print(f"  [Ähnlichkeit: {sim:.2f}]")

    def run_debug_neighbors(raw_inp: str,
                            topn: int = 5,
                            filter_by_label: bool = True):
        sbert_label = sbert_predict([raw_inp])[0]

        print("\n— Debug: ähnlichste Antwortbeispiele —")
        print(f"Eingabe: «{raw_inp}»")
        print(f"SBERT-Label: {sbert_label}")

        q_emb = sbert_model.encode([raw_inp], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, resp_emb)[0]

        candidate_idx = np.arange(len(resp_df))
        if filter_by_label and "label" in resp_df.columns:
            mask = (resp_df["label"].astype(str) == str(sbert_label))
            if mask.any():
                candidate_idx = np.where(mask)[0]

        if len(candidate_idx) == 0:
            print("  (keine Kandidaten im Antwort-Datensatz gefunden)")
            return

        sims_sub = sims[candidate_idx]
        order = np.argsort(-sims_sub)[:min(topn, len(sims_sub))]

        if len(order) == 0:
            print("  (keine passenden Nachbarn gefunden)")
            return

        print(f"\nTop {len(order)} Nachbarn:")
        for rank, local_idx in enumerate(order, start=1):
            idx = candidate_idx[local_idx]
            row = resp_df.iloc[idx]
            sim = sims_sub[local_idx]

            utext = str(row["user_text"])
            ans = str(row["answer_mundart"])
            lbl = row.get("label", "?")
            intent = row.get("intent", "?")
            is_seed = row.get("is_seed", False)

            def _shorten(s, maxlen=80):
                return (s[: maxlen - 1] + "…") if len(s) > maxlen else s

            print(
                f"\n[{rank}] Similarity: {sim:.3f} | "
                f"label={lbl} | intent={intent} | is_seed={is_seed}"
            )
            print("   user_text:     ", _shorten(utext))
            print("   answer_mundart:", _shorten(ans))

    # =====================================================
    # 6) Interaktive CLI
    # =====================================================

    MENU_TEXT = """
Was möchtest du machen?

  1) Sentiment klassifizieren
  2) Next-Word Vorschläge anzeigen
  3) Antwort generieren (inkl. Klassifikation für Label)
  4) Debug: ähnlichste Antwortbeispiele anzeigen
  0) Beenden

Bitte Auswahl eingeben (0-4): 
"""

    print("\nInteraktive Mundart-Demo gestartet.")

    while True:
        try:
            choice = input(MENU_TEXT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAbbruch.")
            break

        if choice == "0":
            print("Beende Interaktion.")
            break

        if choice not in {"1", "2", "3", "4"}:
            print("Ungültige Auswahl. Bitte 0, 1, 2, 3 oder 4 eingeben.\n")
            continue

        try:
            user_text = input(
                "\nBitte Nachricht eingeben (leer zum Abbrechen): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAbbruch.")
            break

        if not user_text:
            print("Keine Eingabe – zurück zum Menü.\n")
            continue

        if choice == "1":
            run_classification(user_text)

        elif choice == "2":
            run_nextword(user_text)

        elif choice == "3":
            cls_info = run_classification(user_text)
            sbert_label = cls_info["sbert_pred"]
            run_answer(user_text, predicted_label=sbert_label)

        elif choice == "4":
            run_debug_neighbors(user_text, topn=5, filter_by_label=True)

        print("\n" + "-" * 60 + "\n")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    # 1) Datensätze erstellen / aktualisieren
    build_base_dataset()
    build_chatpairs_dataset()

    # 2) Training + Interaktive Schleife
    train_and_run()
