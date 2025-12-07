"""
mundart_data.py

- Preprocessing für Schweizerdeutsch-Chat
- Seed-Datensätze (ohne Emojis)
- Chatpair-Datensätze mit Standardantworten
"""

import numpy as np
import pandas as pd
import re
import random

# =========================================================
# Globale Config & Preprocessing
# =========================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

DATA_CSV_BASE = "mundartchat_base.csv"
DATA_CSV_CHATPAIRS = "mundartchat_pairs.csv"

LABEL_ORDER = ["negativ", "neutral", "positiv"]

# Dialekt-Normalisierung (minimal, erweiterbar)
DIALECT_MAP = {
    "nöd": "ned",
    "nid": "ned",
    "ned": "ned",
    "isch": "isch",
    "bisch": "bisch",
    "chan": "chan",
    "chasch": "chasch",
    "chume": "chume",
    "chunsch": "chunsch",
    "huere": "huere",
    "mega": "mega",
    "guet": "guet",
    "geil": "geil",
    "müehsam": "muehsam",
    "müeh": "mueh",
    "müed": "mued",
}

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
USER_RE    = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")

# einfache Token-Definition (ohne Emoji-Specials)
TOKEN_PATTERN = r"(?u)\b[\wäöüÄÖÜß]+\b"


def preprocess_text_chat(t: str) -> str:
    """Einheitliches Preprocessing für Chattexte (ohne Emoji-Sonderlogik)."""
    if t is None:
        return ""
    t = str(t).strip().lower()

    # Platzhalter für URLs, User, Hashtags
    t = URL_RE.sub("<URL>", t)
    t = USER_RE.sub("<USER>", t)
    t = HASHTAG_RE.sub("<HASHTAG>", t)

    # Zahlen normalisieren
    t = re.sub(r"\d+", "<NUM>", t)

    # Mehrfachbuchstaben reduzieren (z.B. "heyyyy" -> "heyy")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)

    # schiefe Apostrophe
    t = re.sub(r"[’´`']", " ", t)

    # Umlaute vereinheitlichen
    t = (t.replace("ä", "ae")
           .replace("ö", "oe")
           .replace("ü", "ue")
           .replace("ß", "ss"))

    # Trenner vereinheitlichen
    t = t.replace("-", " ").replace("/", " ")

    # Dialekt-Normalisierung
    toks = t.split()
    norm_toks = [DIALECT_MAP.get(w, w) for w in toks]
    t = " ".join(norm_toks)

    # alles raus, was kein Wort oder Placeholder ist
    t = re.sub(r"[^\w<>]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


# =========================================================
# 1) Mundart-Chatnachrichten (Seeds, ohne Augmentation)
# =========================================================

# Genau 100 Sätze pro Klasse (negativ/positiv/neutral), natürliche Mundart, ohne Emojis.

EXAMPLES = {
    # =========================================
    # NEGATIV (100)
    # =========================================

    ("negativ", "beschwerde"): [
        "das isch so huere müehsam",
        "ich ha langsam kei nerv meh für dä scheiss",
        "jedes mal s glychi theater",
        "immer lauft öppis schief",
        "ich bi scho wieder enttäuscht",
        "das cha doch nöd wahr sii",
        "nie chunt s so use wie abgmacht",
        "ich ha schnouze voll vo dem",
        "immer wieder di glieche pannen",
        "es isch alles so schlecht organisiert",
        "keim nimmt das richtig ernst",
        "ich muess immer allem noh springe",
        "niemer fühlt sich zuständig",
        "alles verzögert sich eifach",
        "ich chum mir voll verarscht vor",
        "d abmachige werdet nie ihalte",
        "immer nur ausräd statt lösige",
        "es nervt mi nur no",
        "ich bi eifach nöd zfriede",
        "das het nöd im ansatz glappt",
        "so cha mer doch nöd schaffe",
        "ich bi richtig agfrässe",
        "qualitäet isch eifach im keller",
        "es git kei klare infos",
        "keim kommuniziert gscheit",
        "ich ha kei lust meh uf das projekt",
        "jedä schiebt s em andere zue",
        "ich ha s gfuehl, s interessiert niemer",
        "d kundebetreuig isch richtig schwach",
        "ich muess immer alles zwei mol erläbe",
        "so unprofessionell, würkli",
        "mer lernt nüt us de fehler",
        "ich bi es bitzli wütend langsam",
        "das isch mir würkli z vill chaos",
    ],

    ("negativ", "frust"): [
        "ich ha kei bock meh",
        "ich ha null motivation",
        "es isch mir alles z vill",
        "ich bi komplett müed",
        "ich fühl mi richtig usgbrennt",
        "ich ha s gfuehl, es louft immer gege mich",
        "ich weiss nöd, wie lang ich das no cha",
        "ich bi emotional am limit",
        "ich ha scho lang kei pause meh gha",
        "jedä tag fählt sich gliich streng a",
        "ich ha s gfuehl, ich tritt a ort",
        "nüt macht mir grad würkli freud",
        "ich bi nur no am funktioniere",
        "ich bi grad voll im loch",
        "ich wür am liebschte alles hinschmeisse",
        "ich ha kei kraft meh für diskussione",
        "ich bi überlastet mit allem",
        "ich cha nüm richtig schlofe vor lauter chopfkino",
        "ich ha s gfuehl, ich mach alles falsch",
        "ich bi grad würkli dünnhutig",
        "ich han gnueg vo däm stress",
        "ich bin nur no genervt",
        "ich ha kei nerv meh für öppis",
        "alles regt mich uf im momänt",
        "ich cha nüm locker blibe",
        "ich bi verrisse zwüsche allem",
        "ich weiss nöd, wohi mit mir",
        "ich fühl mi eifach leer",
        "ich bruch dringen e change",
        "ich chan nüm so wiiter mache",
        "ich bi chli am abegheitä",
        "ich ha langsam gwungige schultere vom ganze druck",
        "ich ha s gfuehl, niemer versteht das",
    ],

    ("negativ", "hilfe_bitten"): [
        "chan mir bitte öpper s erkläre? ich chum nöd drus",
        "ich bi grad überforderet, chasch mir helfe?",
        "ich weiss nöd, was s nächschti isch, hesch en tipp?",
        "ich bruch dringends en rat",
        "chasch mit mir das schnell durego?",
        "ich versteh s nöd ganz, chasch s no mal erläutere?",
        "ich ha angst, öppis falsch z mache, was meinsch?",
        "ich weiss nöd, wie ich aafange söll",
        "chan ich dir es paar frage stelle dazu?",
        "ich komm mit de situation nöd klar, was söll ich mache?",
        "hesch du scho erfahrig mit dem gha?",
        "ich bi froh um jede hilf, ganz ehrlich",
        "chömer das zäme aaluege?",
        "ich bi nöd sicher, öb ich richtig handle",
        "ich bruch äusseri sicht uf das",
        "chan ich di spöter no mal druf aaluege?",
        "ich verliere grad dr überblick",
        "ich ha s gfuehl, ich steck fescht, was empfiehlsch?",
        "hesch du en vorschlag, wie ich s anders mache chönnt?",
        "ich ha scho vil probiert, nüt het würkli glappt",
        "chan ich uf dini unterstützig zähle?",
        "ich wott das ernst näh, aber ich weiss nöd wie",
        "ich bi unsicher, ob ich überreagiere",
        "chan ich di ächt mit dem belästige?",
        "ich bruch nur chli orientierig im moment",
        "ich ha müeh, das nüchtern aaluege",
        "ich wott nöd alles alei entscheide müesse",
        "ich bi froh, wenn du ehrlich bisch mit mir",
        "wie würdest du an mini stell vorgoo?",
        "ich ha s gfuehl, ich verlüüre d linie",
        "chan ich dich um es ehrlechs feedback bitte?",
        "ich ha langsam kei idee meh",
        "ich wott nöd no meh kaputt mache als scho isch",
    ],

    # =========================================
    # POSITIV (100)
    # =========================================

    ("positiv", "dank"): [
        "merci viu mau, das isch mega lieb vo dir",
        "danke dir, das hät mir sehr gholfe",
        "ich bi dir würkli dankbar",
        "merci, dass du dir d zit gnoh hesch",
        "das isch alles andere als selbstverständlich, danke",
        "danke fürs zuelose",
        "merci fürs dini unterstützig",
        "ich schätz das sehr, was du gmacht hesch",
        "danke, dass du für mich do bisch",
        "merci fürs organisieren",
        "merci für dini müeh",
        "danke für dini ehrlechi wort",
        "ich ha s sehr schön gfunde, merci",
        "danke fürs nachfroge, das bedeutet mir vil",
        "merci, dass du das nöd uf die leicht schulter gnoh hesch",
        "ich bi mega froh, dass du mir gholfe hesch",
        "danke, du hesch mir vil druck gnoh",
        "merci für dini ruäigi art",
        "danke, dass du so geduldig gsi bisch",
        "merci für dä input",
        "ich ha s würkli gmerkt, wie vill müeh du dir gäh hesch",
        "merci, das hät mi ufgstellt",
        "danke, dass du nöd glägig gäh hesch mit mir",
        "danke für dini zuesproch",
        "merci fürs drann denke",
        "danke, dass du das im blick gha hesch",
        "ich wott nur schnell danke säge",
        "merci, du hesch dä tag besser gmacht",
        "danke, dass du so spontan usgheholfe hesch",
        "merci für dini hilfsbereitschaft",
        "ich ha s sehr gschätzt, was du gmacht hesch",
        "danke, dass ich cha uf dich zähle",
        "merci, dass du s mit mir usghalte hesch",
        "danke, du bisch e riesigi unterstützig gsi",
    ],

    ("positiv", "lob"): [
        "das hesch mega guet gmacht",
        "ich bi voll beeindruckt vo dim resultat",
        "du hesch das super umegsetzt",
        "das isch richtig stark worde",
        "ich finds sehr professionell",
        "du chasch würkli stolz sii uf dich",
        "das isch eifach gueti arbeit",
        "ich han riesig freud an dim output",
        "du hesch en super job gmacht",
        "das isch klar und sauber umegsetzt",
        "ich find dini lösig sehr geschickt",
        "du hesch dä nagel uf de chopf troffe",
        "dini idee het mir mega gfalle",
        "das isch besser worde, als ich erwartet ha",
        "ich lern vil vo dim approach",
        "du bisch sehr zuverlässig",
        "ich ha grosses vertroue i dini fähigkeite",
        "de stil vo dir gfallt mir mega",
        "du bisch sehr sorgfältig gsi",
        "ich ha s gfühl, du hesch s voll im griff",
        "dini struktur isch sehr übersichtlich",
        "du bisch extrem konstruktiv mit dä sach umgange",
        "ich find, du machsch das unglaublich guet",
        "dini arbeit het qualität",
        "du bisch es vorbild in däm",
        "ich bin echt fan vo dinere lösig",
        "du hesch es guets händli für so sach",
        "man merkt, dass du vill erfahrig hesch",
        "ich find, du hesch das sehr souverän gmacht",
        "du wirkst sehr kompetent",
        "ich würd das grad so wieder dir überlah",
        "dini leistung isch beeindruckend",
        "ich ha grossi achtig vor dim einsatz",
    ],

    ("positiv", "freud"): [
        "ich freu mi so fescht drüber",
        "ich bi mega happy grad",
        "das macht mi wirklich glücklich",
        "ich ha grad richtig freud",
        "das het mir dä tag verschönert",
        "ich bin voll positiver stimmig",
        "ich chan s fast nöd glaube, wie guet das cho isch",
        "ich bin begeistert, wie sich das entwicklet het",
        "ich han s richtig genosse",
        "ich bi voll entspannt jetz",
        "das git mir viel energie",
        "ich ha s gfuehl, s lauft guet im momänt",
        "ich bin sehr zfriede mit dem",
        "das isch es richtig schöns erlebnis gsi",
        "ich blick grad sehr zuversichtlich i z zukuft",
        "ich bin dankbar für die situation",
        "ich ha s gfühl, ich bi uf em richtige wäg",
        "das gibt mir richtig hoffnig",
        "ich freu mi scho uf s nöchschte mol",
        "ich ha lang nüm so chönne lache",
        "das isch e wohltuendi abwechslig gsi",
        "ich bin froh, dass ich das erlebt ha",
        "ich ha richtig spass gha",
        "ich bi grad total ufgstellt",
        "ich ha s gfuehl, alles füeget sich guet",
        "ich chan nüt negatifs drüber säge",
        "ich würd das sofort wieder so mache",
        "ich bi stolz, wie sich das entwicklet het",
        "ich ha sehr viel freud a dem",
        "das passt im momänt sehr guet für mich",
        "ich bi erleichtert, wie das usecho isch",
        "ich bin innerlich sehr ruhig grad",
        "ich ha s gfuehl, es chunnt no vil guets",
    ],

    # =========================================
    # NEUTRAL (100)
    # =========================================

    ("neutral", "smalltalk"): [
        "hey wie gohts dir?",
        "hoi, was machsch grad so?",
        "na, alles klar bi dir?",
        "was lauft so bi dir?",
        "wie hesch s gha hüt?",
        "wie isch dini wuche bis jetzt gsi?",
        "bisch guet i d wuche startet?",
        "was sind dini plän fürs wucheend?",
        "lange nüm gseh, wie geits?",
        "was treibsch im momänt?",
        "wie isch s im schaffe grad?",
        "hesch grad vil um d ohre?",
        "wie isch s im studi grad?",
        "hesch öppis cools erlebt letschti zit?",
        "wie isch s mit dim projäkt vorwärts gange?",
        "bisch zfriede mit dim alltag im momänt?",
        "wie isch s mit dim gsundheitlich?",
        "hesch dir es bitzeli ruhe gönne chönne?",
        "was luegsch im momänt so für series?",
        "hesch grad öppis guets glese?",
        "wie isch s so im familie-lebe?",
        "hesch no feriä plan?",
        "isch s wätter bi dir au so komisch?",
        "hesch es guets wucheend gha?",
        "wie bisch du i de morge cho?",
    ],

    ("neutral", "orga"): [
        "wänn wür dir passe zum treffe?",
        "chömer s eventuell verschiebe?",
        "passts dir am morn oder am abig besser?",
        "wottsch du online oder vor ort abmache?",
        "wo wäre für dich am praktischste?",
        "wie lang hesch du ungefäär zit?",
        "chunsch direkt oder muesch no öppis erledige?",
        "wänn söll ich dir d details schicke?",
        "wottsch du en kalender-eintrag?",
        "für wann sömer s ungefähr apegge?",
        "mached mer lieber chli kürzer oder länger?",
        "chömer s in zwei teil ufteile, wenn s z vil isch?",
        "wer söll sonst no debi sii?",
        "mached mer eifach es lockers abmache",
        "bin relativ flexibel, wänn hesch du frei?",
        "wänn bruchsch spötestens en beschaid?",
        "wottsch dich zuerst no abspreche mit öpperem?",
        "ich chan diestig und donnschtig, wie isch s bi dir?",
        "treffemer üs grad bim bahnhof?",
        "ich schick dir no es update, sobald ich öppis weiss",
        "gits öppis, wo mir vorher no kläre müend?",
        "chan s au spöter in dr wuche sii, wenn s nöd passt?",
        "mer chönd s au spontan am tag sälber fixiere",
        "söll ich no unterlage vorbereite?",
        "mached mer es doodle, damits eifacher isch?",
    ],

    ("neutral", "frage_info"): [
        "wie funktionieret das genau?",
        "chasch mir kurz erkläre, was ich mues mache?",
        "was meinsch, isch die bescht option?",
        "wie läuft das normalerwiis ab?",
        "was isch dä unterschied zwüsche däne zwei variante?",
        "gits öppis, wo ich im vorus sött wüsse?",
        "mues ich dafür öppis speziells vorbereite?",
        "chan ich das nacher no ändere, falls nötig?",
        "hesch erfahrig mit dem gha?",
        "wie lang gaht das ungefäär?",
        "was sind d vorteil und nachteil devo?",
        "wie hesch du das gmacht, wo du aagfange hesch?",
        "gits öpis, wo ich lieber nöd söll mache?",
        "wo find ich meh infos zu dem?",
        "is das meh öppis für afänger oder nöd?",
        "was würdest du a minere stell mache?",
        "mues ich mich irgendwo registriere dafür?",
        "chan ich es bsüeli experimentiere, ohni vil kaputt z mache?",
        "wie streng isch das im durchschnitt?",
        "brauch ich dafür bestimmti software?",
        "gits öppis, wo ich priorisiera sött?",
        "isch das eimalig oder wiederhole sich das?",
        "chan ich di nöd kurz löchere, wenn ich hänge?",
        "wie handhabed das anderi normalerweise?",
        "was sind dini erfahrige mit dem bis jetzt?",
    ],

    ("neutral", "sonstiges"): [
        "isch mir im fall relativ egal",
        "mir chönd s au eifach mal offe lah",
        "mer lueged, was sich ergit",
        "ich han no kei klare meinig dezue",
        "mir chönd s spontan entscheide",
        "es muess nöd grad hüt sii",
        "ich chan mit beidem lebe",
        "so wichtig isch s im momänt nöd",
        "mir chönd s au no chli beobachte",
        "ich find s weder mega guet no mega schlecht",
        "ich glaub, mer muess s nöd übertreibe mit dem",
        "wenn s nöd klappt, isch au nöd weltuntergang",
        "mer chan s zu gegebener zit wieder aaluege",
        "ich ha jetzt nöd drang, das grad fertig z entscheide",
        "mer chönd s fürs erste so belah wie s isch",
        "ich bi offe für vorschläg",
        "mir mached s so, wie s am eifachste isch",
        "ich ha s nöd pressant mit dem thema",
        "mer chan s gern uf spöter verschiebe",
        "ich wett s nöd komplizierter mache als nötig",
        "wenn du en vorschlag hesch, chömer dä neh",
        "ich nimm s mal zur kenntnis",
        "mir chönd s au schrittwiis aaluege",
        "ich gang mit dem, was sich guet aafühlt",
        "mer müend s nöd uf bieg und bruch entscheide",
    ],
}


# =========================================================
# 2) Mundart-Chatpaare: Default-Antworten
# =========================================================

DEFAULT_ANSWERS_MUNDART = {
    ("negativ", "beschwerde"):
        "Das tönt würklich müehsam. Erzähl mer öppis meh, was genau passiert isch.",

    ("negativ", "frust"):
        "Ich verstand, dass di das grad stresst. Was isch im Momänt am schwierigste für dich?",

    ("negativ", "hilfe_bitten"):
        "Klar, ich probier dir z helfe. Sag mer, wo genau du grad hängt.",

    ("positiv", "dank"):
        "Sehr gärn. Freut mi, dass dir das öppis bringt.",

    ("positiv", "lob"):
        "Schön z ghöre. Du machsch das wirklich guet.",

    ("positiv", "freud"):
        "Mega schön, dass du dich so freusch. Erzähl gern no es bitzeli meh.",

    ("neutral", "smalltalk"):
        "Mir gohts guet, danke. Und dir?",

    ("neutral", "orga"):
        "Klingt guet. Säg mir eifach, wänn s dir am beschte passt.",

    ("neutral", "frage_info"):
        "Gueti Frog. Ich versuechs dir kurz und eifach z erkläre.",

    ("neutral", "sonstiges"):
        "Alles klar, merci für dini Rückmeldung.",
}

DEFAULT_BY_LABEL_MUNDART = {
    "negativ": "Das tönt nöd eifach. Wenn du wotsch, chömer zäme luege.",
    "positiv": "Schön z ghöre. Danke fürs Teile.",
    "neutral": "Alles klar, merci für dini Nachricht.",
}


def get_default_answer_mundart(label: str, intent: str) -> str:
    key = (str(label), str(intent))
    if key in DEFAULT_ANSWERS_MUNDART:
        return DEFAULT_ANSWERS_MUNDART[key]
    return DEFAULT_BY_LABEL_MUNDART.get(str(label), "")


def guess_answer_style(label: str, intent: str) -> str:
    """Grobtyp des Antwortstils (nur als Meta-Info im Datensatz)."""
    if label == "negativ":
        return "empathisch"
    if label == "positiv":
        return "bestärkend"
    if intent in ["frage_info", "orga"]:
        return "erklärend"
    if intent == "smalltalk":
        return "locker"
    return "neutral"


# =========================================================
# 3) Dataset-Build-Funktionen
# =========================================================

def build_base_dataset(out_csv: str = DATA_CSV_BASE) -> pd.DataFrame:
    """Seed-Basisdatensatz (ohne Augmentation) bauen und speichern."""
    rows = []
    for (label, intent), texts in EXAMPLES.items():
        for txt in texts:
            rows.append({
                "text": txt,
                "label": label,
                "intent": intent,
                "is_seed": True,
            })

    base_df = pd.DataFrame(rows)
    base_df = base_df.drop_duplicates(
        subset=["text", "label", "intent"]
    ).reset_index(drop=True)

    # Shuffle für robustere Splits
    base_df = base_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Preprocessed Variante
    base_df["text_clean"] = base_df["text"].astype(str).apply(preprocess_text_chat)

    base_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Neues Basis-DF gespeichert als: {out_csv}")
    print(base_df.head())
    print("\nAnzahl Beispiele total:", len(base_df))
    print("\nKlassenverteilung (label):")
    print(base_df["label"].value_counts())
    print("\nIntent-Verteilung:")
    print(base_df["intent"].value_counts())
    return base_df


def build_chatpairs_dataset(
    in_csv: str = DATA_CSV_BASE,
    out_csv: str = DATA_CSV_CHATPAIRS,
) -> pd.DataFrame:
    """Chatpair-Datensatz (Usertext + Standardantwort) bauen und speichern."""
    df = pd.read_csv(in_csv)
    required_cols = {"text", "label", "intent", "text_clean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Fehlende Spalten in {in_csv}: {missing} – bitte zuerst Basis-Datensatz bauen."
        )

    chatpairs_df = df.copy().rename(columns={
        "text": "user_text",
        "text_clean": "user_text_clean",
    })
    if "is_seed" not in chatpairs_df.columns:
        chatpairs_df["is_seed"] = True

    chatpairs_df["answer_mundart"] = chatpairs_df.apply(
        lambda row: get_default_answer_mundart(row["label"], row["intent"]),
        axis=1,
    )
    chatpairs_df["answer_style"] = chatpairs_df.apply(
        lambda row: guess_answer_style(row["label"], row["intent"]),
        axis=1,
    )
    chatpairs_df["needs_review"] = True

    chatpairs_out = chatpairs_df[[
        "user_text",
        "user_text_clean",
        "label",
        "intent",
        "answer_mundart",
        "answer_style",
        "needs_review",
        "is_seed",
    ]]

    chatpairs_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nNeuer Mundart-Chatpair-Datensatz gespeichert als: {out_csv}")
    print(chatpairs_out.head(10))
    print("\nVerteilung label:")
    print(chatpairs_out["label"].value_counts())
    print("\nVerteilung intent:")
    print(chatpairs_out["intent"].value_counts())
    print("\nAnteil Seeds (is_seed):")
    print(chatpairs_out["is_seed"].value_counts())
    return chatpairs_out


if __name__ == "__main__":
    # Nur Datensätze erstellen
    build_base_dataset()
    build_chatpairs_dataset()
