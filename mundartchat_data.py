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
    # Verneinung / "nur"
    "nid": "ned",
    "noed": "ned",
    "numae": "nume",
    "numeno": "nume no",

    # Formen von "sein"
    "bin": "bi",
    "bini": "bi",
    "ischno": "isch no",
    "sisch": "s isch",

    # Formen von "können"
    "cha": "chan",
    "chani": "chan i",
    "chamer": "chan mer",
    "channsch": "chasch",

    # kommen / gehen
    "chunt": "chunnt",
    "chuntsch": "chunsch",
    "gang": "go",
    "geh": "go",

    # Ausdrücke
    "imfall": "im fall",

    # Adjektive
    "huereguet": "huere guet",
    "hueregael": "huere geil",
    "muesam": "muehsam",
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

EXAMPLES = {
    # =========================================
    # NEGATIV (10 Intents × 10 Bsp)
    # =========================================

    ("negativ", "beschwerde"): [
        "das isch so huere müehsam",
        "ich ha langsam kei nerv meh für dä scheiss",
        "jedes mal s glychi theater",
        "immer lauft öppis schief",
        "ich bi scho wieder enttäuscht",
        "das cha doch nöd wahr sii",
        "es isch alles so schlecht organisiert",
        "keim nimmt das richtig ernst",
        "niemer fühlt sich zuständig",
        "d abmachige werdet nie ihalte",
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
        "jedä tag fählt sich gliich streng a",
        "ich bi grad voll im loch",
    ],

    ("negativ", "hilfe_bitten"): [
        "chan mir bitte öpper s erkläre? ich chum nöd drus",
        "ich bi grad überforderet, chasch mir helfe?",
        "ich weiss nöd, was s nächschti isch, hesch en tipp?",
        "ich bruch dringends en rat",
        "chasch mit mir das schnell durego?",
        "ich versteh s nöd ganz, chasch s no mal erläuterä?",
        "ich weiss nöd, wie ich aafange söll",
        "chan ich dir es paar frage stelle dazu?",
        "ich verliere grad dr überblick",
        "ich ha langsam kei idee meh, was ich no cha probiere",
    ],

    ("negativ", "stress_alltag"): [
        "ich ha s gfuehl, ich chum mit mim alltag nüm nach",
        "alles isch grad uf einisch z vil",
        "ich bi jede tag nur am rennä",
        "mini agenda isch voll und ich ha kei luft meh",
        "i ha kei ahnig, wie ich das alles söll under e huet bringe",
        "s chummt immer no öppis debi und nüt wird weniger",
        "ich ha scho lang kei richtige pause meh gha",
        "jedä tag fängt scho gestresst a",
        "ich ha s gfühl, ich funktionier nume no",
        "i bi am abig eifach komplett dure",
    ],

    ("negativ", "konflikt"): [
        "mir händ scho wieder streit gha",
        "ich ha kei nerv meh für die diskussione",
        "jedä versuecht nume sini sicht dürzdrücke",
        "mir rede ane vorbei und nüt chunt a",
        "i ha s gfuehl, mir verstöhnd üs immer weniger",
        "jedes gspröch eskaliert grad",
        "ich weiss nöd, wie ich mit dem umgah söll",
        "ich bi s pandablet, aber es fählt kei lösig",
        "ich ha kei lust meh uf no meh stress mit däne",
        "ich weiss nöd, ob sich das no flicke laht",
    ],

    ("negativ", "selbstzweifel"): [
        "ich ha s gfuehl, ich bin nöd guet gnue",
        "ich frag mi die ganz zit, öb ich das würklich cha",
        "ich vergleich mi immer mit andere und schneid schlecht ab",
        "ich ha angst, alles nur z versaurä",
        "ich zweifle grad voll a mir selber",
        "ich ha s gfuehl, alli andere chönd s besser als ich",
        "ich ha nöd viel vertroue i mini eigete fähigkeite",
        "jedä chliini fehler macht mi grad fertig",
        "ich ha s gfuehl, ich werd nöd würkli wahrgnuu",
        "ich frag mi, öb ich überhaupt uffem richtige wäg bi",
    ],

    ("negativ", "überforderung_job"): [
        "im schaffe isch s grad viel z vil",
        "mini aufgabe sind imfall nüm machbar i dere zit",
        "ich bi jede tag am limit im büro",
        "ich weiss nöd, wie ich all die projects söll schaffe",
        "i ha s gfuehl, ich brenne bald us",
        "es chunnt immer meh ufträgi debi, aber kei ressourcen",
        "ich ha angst, dä ansprüch nöd z genüege",
        "ich bin im job grad nur am hinterher renne",
        "ich cha nüm abschalte, wenn ich hei chume",
        "ich ha s gfühl, ich gang im job langsam unter",
    ],

    ("negativ", "gesundheit_sorge"): [
        "ich mach mer sorgä um mini gsundheit",
        "ich ha scho lang s gfuehl, öppis stimt nöd mit mir",
        "ich ha immer wieder komische symptöm und weiss nöd, was das isch",
        "ich weiss nöd, öb ich zum arzt söll oder nöd",
        "ich ha angst, dass es öppis ernsthafts isch",
        "mini schlaf isch seit wuche schlecht und das macht mir sorge",
        "ich ha ständig müedigkeit und kenn de grund nöd",
        "ich mach mer au sorgä, öb ich z vill stress ha fürs herz",
        "ich ha s gfuehl, mis imunsystem isch im keller",
        "ich weiss nöd, wie lang das no so cha wiiter gah",
    ],

    ("negativ", "enttäuschung_beziehung"): [
        "ich bi grad recht enttäuscht vo dere person",
        "ich ha s gfuehl, ich bi nöd priorität für sie",
        "mir händ abgmacht gha und sie händ s wieder nöd ernst gnoh",
        "mini erwartige sind grad voll nöd erfüllt worde",
        "ich ha sehr vil i die beziehig investiert und wenig zrugg übercho",
        "ich ha s gfuehl, ich werd nöd würklich glosä",
        "i bi nöd sicher, öb mer no am gliiche ort sind",
        "ich ha s vertroue chli verlore",
        "ich bin traurig, wie sich das entwicklet het",
        "ich ha s gfuehl, ich bedeut nöd meh so vil wie früher",
    ],

    ("negativ", "einsamkeit"): [
        "ich fühl mi imfall recht einsam im momänt",
        "au wenn lüüt um mich ume sind, fühl ich mi allei",
        "ich ha s gfuehl, niemer versteht mi würkli",
        "ich weiss nöd, mit wem ich cha richtig drüber rede",
        "alli händ irgendwie ihre gruppe, nur ich nöd",
        "ich würd mir meh nächi wünsche, aber ich weiss nöd, wie",
        "ich han vil kontakt online, aber weni, wo sich echt aafühlt",
        "ich vermisse s gfühl, würklich ufgnoh z sii",
        "ich fühl mi imfall schono längers isoliert",
        "ich ha s gfuehl, ich zieh mi immer meh zrugg",
    ],

    # =========================================
    # POSITIV (10 Intents × 10 Bsp)
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
        "ich find dini lösig sehr geschickt",
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
        "ich ha s richtig genosse",
        "das git mir viel energie",
    ],

    ("positiv", "erfolg"): [
        "ich ha mini prüefig bstange, bin voll erleichtert",
        "s projekt isch mega guet usecho",
        "ich ha es wichtigs ziel erreicht, wo ich lang dra gschaffet ha",
        "mini müeh hät sich würkli glohnt",
        "ich ha es komplizierte thema endlich checkt",
        "ich ha feedback übercho, dass mini arbeit sehr guet isch gsi",
        "ich bi e schritt wiiter cho, als ich denkt ha",
        "ich bi stolz, dass ich das durchzoge ha",
        "ich ha en schöne erfolg im job gha",
        "ich ha s gfühl, ich mach imfall fortschritt",
    ],

    ("positiv", "erleichterung"): [
        "ich bin so erleichtert, dass das jetzt um isch",
        "ich ha lang sorgä gha und jetzt isch s doch guet cho",
        "es isch e riesen stei vom härz gfalle",
        "endlich isch d entscheidig do, und s isch besser cho als erwartet",
        "ich ha s gfühl, ich chan wieder freier atme",
        "ich bi froh, dass sich das nöd so schlimm entwicklet het",
        "ich ha lang druf gwartet und jetzt isch s eifach e guets gfühl",
        "mini angst het sich zum gliick nöd bestätiget",
        "ich ha etz wieder chli ruhe im chopf",
        "ich bi froh, dass ich das chli hinter mir han",
    ],

    ("positiv", "stolz"): [
        "ich bi würkli stolz uf mich grad",
        "ich ha öppis gstemt, wo ich mir selber nöd zuegtraut ha",
        "es isch schön z gseh, wie wiit ich cho bi",
        "ich ha s gfuehl, ich cha zurecht chli stolz sii",
        "ich ha hart drfür gschaffet und jetzt zahlts sich us",
        "ich bi zfriede mit dem, wie ich mit dere situation umgange bi",
        "ich bi stolz, dass ich drblibe bi, au wenn s schwierig gsi isch",
        "ich ha min weg gmacht, Schritt für Schritt",
        "ich cha uf das resultat ohni schlechts gfühl luege",
        "ich bin stolz, dass ich mini werte nöd verloo ha",
    ],

    ("positiv", "verbindung_freunde"): [
        "mir händ en extrem gmüetliche abig gha zäme",
        "ich bi mega froh um mini fründschaftä",
        "mir händ so vil glacht, das het richtig guet ta",
        "ich fühl mi grad voll ufgnoh i dere gruppe",
        "ich ha s gfühl, mer verstönd sich immer besser",
        "mir händ offe chönne rede über ernsti sache",
        "ich schätz es mega, so mensche um mich z ha",
        "ich ha lang nüm so e ehrlechs gspröch gha",
        "ich bin dankbar für die nächi, wo mer händ",
        "ich ha s gfühl, ich bi nöd alei mit mine themä",
    ],

    ("positiv", "motivation"): [
        "ich bi imfall grad voll motiviert",
        "ich ha richtig lust, öppis azpacke",
        "ich ha en guete drive im momänt",
        "ich ha s gfühl, jetzt isch en guete zeitpunkt zum starte",
        "ich ha so vil ideä im chopf, wo ich umsetze wott",
        "ich spür grad richtig energie i mir",
        "ich ha s gfühl, ich chan grad vil bewege",
        "ich freu mi sogar uf d nägschti aufgabä",
        "ich ha imfall grad vill bock zum dranzbliibe",
        "ich bin parat, en schritt wiiter z gah",
    ],

    ("positiv", "vorfreude"): [
        "ich freu mi mega uf d feriä",
        "ich ha so vorfreud uf dä termin",
        "ich chan s fast nüm abwarte, bis es so wiit isch",
        "ich bi scho am plane, wie das chönnt werde",
        "ich hab e richtig guets gfühl für das, was chunt",
        "ich zähl fasch d tägli bis dr tag chunnt",
        "ich freu mi uf d lüt, wo ich denn wieder gseh",
        "ich han lang uf dä moment gwartet",
        "ich ha s gfühl, das wird öppis schöns",
        "die vorfreud git mir grad viel energie",
    ],

    ("positiv", "zufriedenheit_alltag"): [
        "ich bi imfall im momänt recht zfriede mit mim alltag",
        "es lauft nöd perfekt, aber es stimmt für mich",
        "ich ha e gueti balans zwüsche pause und leistung gfunde",
        "ich ha s gfühl, s passt so im momänt",
        "ich chan mit dim, wie s grad isch, guet lebe",
        "mini rutine tuet mir guet",
        "ich bi froh, dass nöd jede tag drama isch",
        "ich ha s gfühl, ich ha mini säch im griff",
        "ich chan au chli ruhe gniesse zwüschet dur",
        "ich bi dankbar für die stabilität, wo ich grad ha",
    ],

    # =========================================
    # NEUTRAL (10 Intents × 10 Bsp)
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
    ],

    ("neutral", "orga"): [
        "wänn wür dir passe zum treffe?",
        "chömer s eventuell verschiebe?",
        "passts dir am morn oder am abig besser?",
        "wottsch du online oder vor ort abmache?",
        "wo wäre für dich am praktischste?",
        "wie lang hesch du ungefäär zit?",
        "wänn söll ich dir d details schicke?",
        "wottsch du en kalender-eintrag?",
        "für wann sömer s ungefähr apegge?",
        "chan s au spöter i de wuche sii, wenn s nöd passt?",
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
    ],

    ("neutral", "status_update"): [
        "kurz es update: es lauft imfall solid, aber langsam",
        "ich bi mittendrin im projektschub",
        "es isch no nöd alles fertig, aber ich bi druf und dra",
        "ich ha s gröbschte gmacht, aber es brucht no feinarbeit",
        "ich bi no nöd dort, wo ich wott sii, aber ufem weg",
        "bis jetzt isch no nüt eskaliert, es lauft ruhig",
        "ich han scho es bitzeli fortschritt gmacht",
        "im moment isch s chli viel, aber im rahme",
        "es isch imfall alles no im gange",
        "ich gib dir wieder bschaid, sobald öppis neus isch",
    ],

    ("neutral", "planung_langfristig"): [
        "ich versueche grad mini längfristige plän z ordne",
        "ich überleg mer, wie s die nächschte jahr sött usegseh",
        "ich weiss no nöd genau, wohi, aber ich wott chli struktur ine bringe",
        "ich probier chli vorus z plane statt nume spontan",
        "ich möcht mir e chli klari ziel setze fürs nöchschte jahr",
        "ich bin am luege, wie sich jobb und privat chönd verbinde lah",
        "ich wott eigentli chli meh ruhe i mini plän ine bringe",
        "ich mues no uussortiere, was mir würkli wichtig isch uf d längi",
        "ich denk grad vil drüber nach, was ich langfristig wott",
        "ich probier Schritt für Schritt öppis z verändere, nöd alles uf einisch",
    ],

    ("neutral", "feedback_sachlich"): [
        "nur chli feedback: es isch übersichtlich, aber chönnt no klarer sii",
        "ich find s grundsätzlich guet, aber es het no paar punkt zum verbessere",
        "vo minere sicht her funktioniert s, aber d doku chönnt besser sii",
        "ich finds sachlich okay umgsetzt, d struktur isch logisch",
        "ich würd eifach no chli meh erläuterige erwarte",
        "mini rückmeldung wär, dass d zeitplän chli optimiert chönd wärde",
        "inhaltlich isch s stabil, optisch chan mer no drann schaffe",
        "es isch guet lösbar, aber d anleitig chönnt knapp formuliert sii",
        "ich find s fair, aber chli streng i dere form",
        "vo mir us es nüchternes, aber positives feedback",
    ],

    ("neutral", "hobby_talk"): [
        "ich bi grad wieder vil am gamä i minere freizit",
        "ich go gern jogge, wenn ich chli kopf lüfte möcht",
        "ich ha wieder aagfange meh z läse",
        "ich bi im moment voll im kochfilm",
        "ich lueg viel series, wenn ich abends am abchille bi",
        "musik lose hilft mer mega zum abschalte",
        "ich bin grad am übe ufem instrument",
        "ich ha neu mit fotografie aagfange",
        "ich bi gern drausse underwägs, wenn s wätter mitmacht",
        "ich versueche meh kreativs zeug i mini freizit ine z bringe",
    ],

    ("neutral", "news_teilen"): [
        "hesch gseh, was grad i de nöi passiert isch?",
        "ich ha grad es interessants artikel glese über das thema",
        "es het wieder es update zue dere sach geh",
        "imfall, es git jetzt es neus feature bi dere app",
        "ich ha ghört, dass sich bi dere regel öppis gänderet het",
        "es isch grad en spannende entwicklung i dem bereich am laufe",
        "ich bi über en beitrag gstolpert, wo guet z dem passt",
        "grad in de nöi isch vil diskussion zu dem thema",
        "es het es neus projekt, wo spannend chönnt sii",
        "ich ha gseh, dass d termi sich chli gschobe händ",
    ],

    ("neutral", "technik_support"): [
        "mini app spinnt scho wieder und ich weiss nöd, wieso",
        "ich chum nöd drus mit däm update",
        "mis wlan macht grad komische sache",
        "ich weiss nöd, wie ich das programm söll installiere",
        "dä login will eifach nöd funktioniere",
        "ich ha s gfuehl, mis handy isch extrem langsam worde",
        "ich bechum ständig fehlermeldig und check nöd, was s heisst",
        "ich weiss nöd, welches kabel wohi ghört",
        "ich ha öppis verstellt i de einstellige und find s nüm",
        "ich bruch nur chli hilf, zum das technisch wieder zum laufe bringe",
    ],
}



# =========================================================
# 2) Mundart-Chatpaare: Default-Antworten
# =========================================================

DEFAULT_ANSWERS_MUNDART = {
    # NEGATIV
    ("negativ", "beschwerde"):
        "Das tönt würklich müehsam. Erzähl gern no chli meh, was genau di stört.",
    ("negativ", "frust"):
        "Ich verstand, dass dich das nervt. Was isch im Momänt am müehsamste für dich?",
    ("negativ", "hilfe_bitten"):
        "Ich probier gern z helfe. Sag eifach, wo du grad hängsch.",
    ("negativ", "stress_alltag"):
        "Alltag cha schnell z vil wärde. Womit fangemer am beschte aa, zum s chli z ordne?",
    ("negativ", "konflikt"):
        "Konflikt sind nöd eifach. Was isch dir da besonders wichtig i dä situation?",
    ("negativ", "selbstzweifel"):
        "Es isch okay, wenn mer sich mal unsicher fühlt. Worin hesch du grad am meiste zweifel?",
    ("negativ", "überforderung_job"):
        "Jobstress cha würklich belaste. Was belastet di im momänt am stärkschte?",
    ("negativ", "gesundheit_sorge"):
        "Gsundheitssorgä nimmt mer sich zu herz. Was macht dir im momänt am meiste sorge?",
    ("negativ", "enttäuschung_beziehung"):
        "Enttäuschige tue weh. Wenn du wotsch, chasch chli erläutere, was passiert isch.",
    ("negativ", "einsamkeit"):
        "Sich einsam fühle isch nöd eifach. Schön, dass du drüber redsch. Wie fült sich das grad a für dich?",

    # POSITIV
    ("positiv", "dank"):
        "Sehr gärn. Freut mi, wenn es dir öppis bringt.",
    ("positiv", "lob"):
        "Schön z ghöre, dass das so guet cho isch. Mach eifach so wiiter.",
    ("positiv", "freud"):
        "Mega schön, dass dich das so freut. Erzähl gern no meh drüber, wenn du wotsch.",
    ("positiv", "erfolg"):
        "Gratuliere, das isch en schöne erfolg. Nimm dir ruhig en moment, zum stolz z sii.",
    ("positiv", "erleichterung"):
        "Guet z ghöre, dass s sich jetzt leichter aafühlt. Was hät dir am meiste gholfe?",
    ("positiv", "stolz"):
        "Du darfsch stolz sii uf dich. Was gfallt dir selber am beschte a dim weg bis da?",
    ("positiv", "verbindung_freunde"):
        "Schön, dass du so mensche um dich hesch. Solchi momänt tue mega guet.",
    ("positiv", "motivation"):
        "Das tönt nach vill energie. Wie wotsch die motivation für dich nutze?",
    ("positiv", "vorfreude"):
        "Vorfreud isch öppis Schöns. Uf was freusch di am meiste?",
    ("positiv", "zufriedenheit_alltag"):
        "Schön, dass sich dis läbe im momänt stimmig aafühlt. Was wotsch gern so biibe lah?",

    # NEUTRAL
    ("neutral", "smalltalk"):
        "Mir gohts guet, merci. Und dir so im alltag?",
    ("neutral", "orga"):
        "Klingt guet. Säg eifach, was dir am beschte passt, denn chömer s so abmache.",
    ("neutral", "frage_info"):
        "Gueti frog. Ich versuech s dir möglichst eifach z erläuterä.",
    ("neutral", "sonstiges"):
        "Alles klar, merci für dini rückmeldung.",
    ("neutral", "status_update"):
        "Merci fürs update. Wenn sich öppis ändert, chasch eifach wieder schriebe.",
    ("neutral", "planung_langfristig"):
        "Okay, denn chömer gern zäme luege, wie mer das langfristig cha aagle.",
    ("neutral", "feedback_sachlich"):
        "Danke fürs sachlich feedback. Das hilft, s besser iigschätze z chönne.",
    ("neutral", "hobby_talk"):
        "Klingt spannend. Erzähl gern no meh über dini hobbys, wenn du lusch hesch.",
    ("neutral", "news_teilen"):
        "Danke fürs teile. Spannend, was grad alles lauft.",
    ("neutral", "technik_support"):
        "Okay, luegemer zäme druf. Was genau funktioniert im momänt nöd so, wie s sött?",
}

DEFAULT_BY_LABEL_MUNDART = {
    "negativ": "Das tönt nöd eifach. Wenn du wotsch, luegemer zäme, was dir hälfe chönnt.",
    "positiv": "Schön z ghöre. Danke, dass du das teilsch.",
    "neutral": "Alles klar, merci für dini nachricht.",
}



def get_default_answer_mundart(label: str, intent: str) -> str:
    key = (str(label), str(intent))
    if key in DEFAULT_ANSWERS_MUNDART:
        return DEFAULT_ANSWERS_MUNDART[key]
    return DEFAULT_BY_LABEL_MUNDART.get(str(label), "")


# =========================================================
# 3) Augmentation
# =========================================================

SYNONYM_MAP = {
    # eher negativ / belastend
    "müehsam": ["streng", "nervig", "anstrengend"],
    "müed": ["fix und fertig", "kaputt", "voll dure"],
    "stress": ["druck", "belastig"],
    "usgbrennt": ["am limit", "komplett dure"],

    # eher positiv
    "mega": ["huere", "richtig", "extrem"],
    "guet": ["tiptop", "lässig", "solid"],
    "freud": ["spass", "gaudi"],
    "happy": ["glücklich", "ufgstellt"],

    # eher neutral / weich
    "okay": ["im rahme", "eigentlich guet", "so lala"],
    "egal": ["nöd so wichtig", "nöd tragisch"],
}

GREETINGS = ["hey", "hoi", "ey", "jo", "sali", "hallo du"]

NEG_TAILS = [
    "gopfertami",
    "gopfriedstutz",
    "so en seich",
    "so en käse",
    "so en scheiss",
    "eifach müehsam",
    "richtig müehsam",
    "zum schreiä",
    "zum verseckä",
    "zum d wand ufe laufä",
]

POS_TAILS = [
    "mega guet",
    "richtig schön",
    "so geil",
    "voll cool",
    "eifach stark",
    "richtig lässig",
    "eifach nur schön",
    "vo härze her guet",
    "zum sich freue",
    "zum dankbar sii",
]

NEU_TAILS = [
    "so im fall",
    "mal luege",
    "mal gseh",
    "denke ich",
    "wür ich säge",
    "isch okay so",
    "passt so für mich",
    "chömer so lah",
    "für s erscht mal guet",
    "mached mer so imfall",
]

GOODBYES = [
    "lg",
    "liebi grüess",
    "gruessli",
    "bis bald",
    "ciao",
    "tschüss",
    "mach s guet",
    "schönä tag no",
    "schöns wucheend",
    "bis spöter",
]


def replace_with_synonym_once(txt: str) -> str:
    """Ersetzt höchstens ein Wort durch ein Synonym (falls vorhanden)."""
    toks = txt.split()
    if not toks:
        return txt

    candidate_indices = [
        i for i, w in enumerate(toks)
        if w in SYNONYM_MAP
    ]
    if not candidate_indices:
        return txt

    idx = random.choice(candidate_indices)
    word = toks[idx]
    synonym = random.choice(SYNONYM_MAP[word])
    toks[idx] = synonym

    return " ".join(toks)


def augment_with_style(txt: str, label: str) -> str:
    """Augmentation: optionale Begrüssung, optional Synonym, Sentiment-Tail, Verabschiedung."""
    toks = txt.split()
    if not toks:
        return txt

    # 1) optional Begrüssung vornedran
    if random.random() < 0.25:
        greeting = random.choice(GREETINGS)
        toks = [greeting + ","] + toks

    out = " ".join(toks)

    # 2) optional genau ein Synonym ersetzen
    if random.random() < 0.4:
        out = replace_with_synonym_once(out)

    # 3) optional Anhängsel je nach Sentiment
    r = random.random()
    if label == "negativ" and r < 0.3:
        out = out + " " + random.choice(NEG_TAILS)
    elif label == "positiv" and r < 0.3:
        out = out + " " + random.choice(POS_TAILS)
    elif label == "neutral" and r < 0.3:
        out = out + " " + random.choice(NEU_TAILS)

    # 4) optional Verabschiedung ans Ende
    if random.random() < 0.25:
        out = out + " " + random.choice(GOODBYES)

    return out


def upsample_to_target_per_label(
    df: pd.DataFrame,
    target_per_label: int = 500,
) -> pd.DataFrame:
    """Erweitert den Datensatz pro label via Augmentation (Begrüssung, Synonyme, Tails, Verabschiedung)."""
    all_rows = [df]
    for label in df["label"].unique():
        cur = df[df["label"] == label]
        needed = target_per_label - len(cur)
        if needed <= 0:
            continue

        records = cur.to_dict("records")
        generated = []
        i = 0
        while len(generated) < needed:
            base_row = records[i % len(records)].copy()
            base_row["is_seed"] = False
            base_row["text"] = augment_with_style(base_row["text"], label=label)
            generated.append(base_row)
            i += 1

        all_rows.append(pd.DataFrame(generated))

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # text_clean neu berechnen
    out["text_clean"] = out["text"].astype(str).apply(preprocess_text_chat)
    return out



# =========================================================
# 4) Dataset-Build-Funktionen
# =========================================================

def build_base_dataset(
    out_csv: str = DATA_CSV_BASE,
    target_per_label: int | None = 500,
) -> pd.DataFrame:
    """Seed-Basisdatensatz bauen, optional per label auf Zielgrösse augmentieren."""
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

    if target_per_label is not None:
        base_df = upsample_to_target_per_label(
            df=base_df,
            target_per_label=target_per_label,
        )
    else:
        # falls du mal ohne Augmentation fahren willst
        base_df["text_clean"] = base_df["text"].astype(str).apply(preprocess_text_chat)

    base_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Neues Basis-DF gespeichert als: {out_csv}")
    print(base_df.head())
    print("\nAnzahl Beispiele total:", len(base_df))
    print("\nKlassenverteilung (label):")
    print(base_df["label"].value_counts())
    print("\nIntent-Verteilung:")
    print(base_df["intent"].value_counts())
    print("\nAnteil Seeds (is_seed):")
    print(base_df["is_seed"].value_counts())

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
    chatpairs_df["needs_review"] = True

    chatpairs_out = chatpairs_df[[
        "user_text",
        "user_text_clean",
        "label",
        "intent",
        "answer_mundart",
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
