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

# Dialekt-Normalisierung (minimal, nur Formen, die im Datensatz vorkommen)
DIALECT_MAP = {
    # Verneinung: "nöd" -> "ned"
    "noed": "ned",
    # "ich bin" -> "ich bi"
    "bin": "bi",
    # "ich cha" -> "ich chan"
    "cha": "chan",
    # fester Ausdruck
    "imfall": "im fall",
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
        "das isch eifach nur nervig, ehrlich",
        "jedes mol läuft öppis schief, ich cha nüm",
        "ich ha langsam schnauze voll vo dem chaos",
        "ich werd no wahnsinnig mit dem ganze hin und her",
        "alles zieht sich so unglaublich i d länge",
        "es chunt mer vor, als nähm das niemert ernst",
        "ich bi so müed vo dem ständige stress",
        "nüt funktioniert, wie s eigentlich sött",
        "jedes problem macht grad no meh problem",
        "ich ha s gefühl, kei mensch denkt das richtig dür",
        "ich bi gopfertami so enttäuscht",
        "ich würd mir eifach mol öppis wünschen, wo klappt",
        "immer mues irgendöppis daneben gah",
        "es regt mi nur no uf",
        "mini geduld isch definitiv am end",
        "ich cha das theater nüm bsorge",
        "ich ha kei nerv meh für so unorganisierts zeug",
        "es chunnt mer vor wie pure inkompetenz",
        "jedes mol müemer vo vorne aafange",
        "ich bi scho wieder am motze, aber s isch würkli müehsam",
        "wieso isch das immer so kompliziert?",
        "ich ha s gfühl, d sache werde eher schlechter als besser",
        "d ganze situation isch nur frustrierend",
        "ich mues mi jedes mol über öppis anders ufrege",
        "chönt mer das nöd eifach mol gscheit mache?",
        "ich verstah nöd, wieso das so schwierig isch",
        "ich glaub langsam nüm, dass das jemals richtig lauft",
        "jedes update macht s eher schlimmer",
        "immer wenn ich hoffig ha, passiert öppis blöds",
        "ich bi eifach nur no genervt vo dem",
    ],


    ("negativ", "frust"): [
        "ich weiss nöd, wie lang ich das no pack",
        "ich fühl mi grad komplett fertig",
        "ich cha mi nüm motiviere für öppis",
        "alles isch mir grad z viel",
        "ich würd am liebschte alles hinschmeisse",
        "ich ha s gfühl, ich dreh mi im churzschlus",
        "ich ha null energie im momänt",
        "ich kämpf und kämpf, aber es bringt nüt",
        "ich weiss nöd, wo ich ahfange söll",
        "ich fühl mi wie blockiert",
        "es isch alles so schwer grad",
        "ich bi mega dünnhuutit im momänt",
        "ich ha s gfühl, alles lauft gegen mi",
        "nüt gelingt mir richtig",
        "ich würd gärn pause mache, aber s gaht nöd",
        "ich ha kei plan meh, wie wiiter",
        "ich wott eifach chli ruhe ha",
        "ich fühl mi überrollt vo allem",
        "ich cha nüm so tuen, als wär alles easy",
        "ich bi voll im kopfchaos",
        "ich wünschte, ich chönnt endlich abschalte",
        "ich weiss nöd, wieso ich grad so struggl",
        "ich würd gern positiv bliibe, aber s goht nöd",
        "ich fühl mi mega lost im momänt",
        "jede chliini sache stresst mi übertriebe",
        "ich ha s gfühl, ich mach nüt richtig",
        "ich würd so gärn es reset drücke",
        "ich fühl mi brutal überfodert",
        "ich ha mini emotionen nüm im griff",
        "ich bruch dringend chli entlastig",
    ],


    ("negativ", "hilfe_bitten"): [
        "chasch mir bitte erklärä, wie das goht? ich chum nöd drus",
        "ich bruch öpper, wo mir das Schritt für Schritt zeigt",
        "ich bi komplett lost, chasch mir helfe?",
        "sorry, aber ich verstand das gar nöd",
        "hesch chli zit, mir das kurz z erkläre?",
        "ich weiss nöd, wo ich söll aafange",
        "chasch mir en tipp geh, wie ich das löse?",
        "ich versteh s nöd ganz, chasch das anders usdrücke?",
        "ich bruuch e chli guidance",
        "chasch mir das vormache, bitte?",
        "ich han paar frage – hesch kurz zeit?",
        "ich versuech s, aber ich schnapp s nöd",
        "wär mega, wenn du mir da chönntisch ushelfe",
        "ich bi überforderet, chasch es mir erläuterä?",
        "ich weiss nöd, öb ich s richtig mache",
        "chöntisch mir e beispiel geh?",
        "ich würd s gern checke, aber mir fehlt dä überblick",
        "chasch mir s grob zämäfasse?",
        "ich bi unsicher, chasch du drüber luege?",
        "ich bruch nur es chli unterstützig",
        "chasch mir s laufe kurz erklärä?",
        "ich bi komplett dra vorbeigfahre – chasch mir helfe?",
        "ich ha dä schritt nöd verstanden, chasch?",
        "ich probiers, aber es klappt nöd, chasch du?",
        "ich bruch en hinwiis, wie das funktioniert",
        "chasch du mir s vormache, wie mer das richtig macht?",
        "ich kämpf grad, chasch es mir bitte erläutere?",
        "chasch mir s eifacher erkläre?",
        "ich bruch en zweite blick druf",
        "ich kapier s nöd, chasch du mir helfe?",
    ],


    ("negativ", "stress_alltag"): [
        "alles isch grad mega hektisch",
        "ich renn de ganze tag ume",
        "ich ha kei minute für mich selber",
        "es tuet sich immer meh ufstaple",
        "ich weiss nöd, wie ich alles in de tag packe söll",
        "ich ha s gfühl, ich bin nume am funktioniere",
        "mini agenda isch komplett überfüllt",
        "ich chum kaum no zum durchatme",
        "jedes mol chunnt öppis neus debi",
        "ich bruch dringend ruhe",
        "ich bi scho am morn gestresst",
        "ich bin nur am rölle de ganz tag",
        "es schlägt mir langsam uf s gemüt",
        "ich ha kei puffer meh",
        "ich ha s gfühl, ich ha kei kontrolle meh",
        "ich bi immer e schritt z spoot",
        "ich ha s gfühl, ich lebe nume no im stressmodus",
        "ich bi nüm fähig, richtig abzuschalte",
        "ich würd gern mal e chli ruhe ha",
        "ich ha grad vil z vil uf em tisch",
        "ich han kei kopf frei für öppis anders",
        "ich bin komplett überlastet",
        "ich weiss nöd, wie ich das tempo lang cha halte",
        "ich fühl mi nur no erschöpft",
        "ich bruch dringend entschleunigung",
        "ich bi am limit vo minere kapazität",
        "ich würd gern mol e tag ohni to-do-lischt ha",
        "ich kann mich grad uf nüt konzentriere",
        "ich ha kei erholig zwüschet dur",
        "ich kämpfe täglich mit dem stresslevel",
    ],


    ("negativ", "konflikt"): [
        "mir händ scho wieder en riesen streit gha",
        "ich ha kei nerv meh für das ständige diskutiere",
        "jedes gspröch eskaliert instant",
        "mir verstöhnd üs gar nüm",
        "alles driftet immer i d falschi richtig",
        "ich ha s gfühl, mir reded komplett ane vorbei",
        "ich weiss nöd, wie mer das löse sött",
        "ich cha nüm immer dr vermittler spiele",
        "mini geduld isch absolut am limit",
        "mer chönd nöd mol sachlich über s thema rede",
        "es isch wie en endlosi schleife vo missverständnis",
        "ich fühl mi nöd ernst gno i däm gspröch",
        "ich bi müed vo dem ständige hin und her",
        "jedes mol chunnt öppis neus, wo alles schwieriger macht",
        "ich ha s gfühl, ich mues immer nachgeh",
        "mer finded kei gemeinsame basis meh",
        "ich weiss nöd, öb mer überhaupt no ufem gliiche level sind",
        "min standpunkt wird ständig verdreht",
        "ich fühl mi nöd respektiert",
        "ich weiss nöd, was ich no söll probiere",
        "d situation isch mega emotional ufglade",
        "ich bi erschöpft vo all den diskussione",
        "es stimmt eifach nüt meh zwüschet üs",
        "ich ha angst, dass mer das nüm chönd flicke",
        "mer reagiered beidi übertriebe schnell",
        "ich würd gärn sachlich bliibe, aber s gelingt nöd",
        "ich weiss nöd, wohi das führt",
        "niemer loset richtig zue",
        "jedes mal endet s im frust",
        "ich weiss grad nöd, wie mer da wieder usechöme",
    ],


    ("negativ", "selbstzweifel"): [
        "ich ha s gfühl, ich bi nöd guet gnue",
        "ich zweifle grad extrem a mir selber",
        "ich weiss nöd, öb ich dä challenge würkli packe cha",
        "ich fühl mi häufig ungenüeg",
        "ich ha angst, alles falsch z mache",
        "ich vergleiche mi immer und schneid schlecht ab",
        "ich bremse mi selber ständig uus",
        "ich weiss nöd, öb ich würkli öppis wert bi",
        "ich fühl mi wie en riesigs fragezeiche",
        "ich bi nöd stolz uf das, was ich mach",
        "ich ha s gefühl, ich enttäusch alli",
        "ich ha angst, dass ich nüm vorwärts chume",
        "ich glaub nöd an mini eigete fähigkeit",
        "ich fühl mi wie en versager",
        "ich weiss nöd, wieso ich so a mir zweifle",
        "jedes chliini problem bringt mi zum grüble",
        "ich würd gärn meh selbstvertroue ha",
        "ich ha s gfühl, ich mach alles nur halb guet",
        "ich weiss nöd, öb ich gut genug bi für dini erwartige",
        "ich fühl mi oft fehl am platz",
        "ich weiss nöd, wieso ich mir selber so im weg stah",
        "ich mach mir vil z viel druck",
        "ich fühl mi mega verletzlich",
        "ich weiss nöd, öb ich s verdient ha",
        "ich ha angst vor fehler und drum probier ich nüüt uus",
        "ich fühl mi extrem unsicher im momänt",
        "ich würd gern besser zu mir selber sii",
        "ich ha s gfühl, ich bi nöd gnue für ander lüt",
        "ich kämpf viel mit minere innere kritikerstimme",
        "ich weiss grad nöd, wie ich us däm loch uschume",
    ],


    ("negativ", "überforderung_job"): [
        "im job isch grad extrem viel los, ich cha nüm",
        "ich ha s gfühl, ich renn de ganze tag hinterher",
        "z vil ufträgi uf einisch, ich pack s nöd",
        "ich weiss nöd, wie ich das bewältige söll",
        "mini arbeit wächst mir über d chopf",
        "ich ha kei ressourcen meh für das",
        "ich bruch imfall dringend en break",
        "ich ha s gfühl, ich brenn us",
        "es chunnt immer meh debi und nüt wird weniger",
        "ich stoss grad a mini grenzen",
        "ich cha nüm klar denke vor lauter stress",
        "mini to-do-lischt macht mi fertig",
        "ich würd gärn konzentriert schaffe, aber s gaht nöd",
        "ich ha angst, dass ich s nöd pack",
        "ich würd am liebschte alles kurz stoppe",
        "ich bin extrem unter druck",
        "mini arbeit überforderet mi grad brutal",
        "ich bruch unterstützig, aber es chunnt kei",
        "ich ha immer eng termini, das stresst brutal",
        "ich cha nüm abschalte nach em schaffe",
        "ich fühl mi überrollt vo de ufgabe",
        "ich ha s gfühl, alli erwarted z vil vo mir",
        "mini motivation isch am sinke",
        "ich ha angst, dass ich öppis versaue",
        "ich würd gärn usgleiche finde, aber schaff s nöd",
        "ich glaub, ich bin am absolute limit",
        "ich ha kei kontrolle meh über mini workload",
        "ich chumm nüm zum pause mache",
        "ich bi erschöpft vo dem jobstress",
        "ich weiss nöd, wie lang ich das no so chume",
    ],


    ("negativ", "gesundheit_sorge"): [
        "ich mach mer grad mega sorgä um mini gesundheit",
        "ich weiss nöd, wieso ich mich ständig komisch fühl",
        "ich ha s gfühl, öppis stimmt nöd mit mir",
        "ich ha symptome, wo mi verunsichere",
        "ich ha angst, dass es öppis ernsthafts isch",
        "ich bin unsicher, öb ich zum arzt söll",
        "ich bi ständig müed, das macht mi fertig",
        "ich fühl mi körperlich instabil",
        "ich ha immer wieder unruh im körper",
        "ich mach mi verrückt mit gedanke",
        "ich ha s gfühl, mis imunsystem spinnt",
        "ich weiss nöd, wo d ursach isch",
        "ich ha angst, dass es schlimmer wird",
        "ich weiss nöd, wie ich das einschätze söll",
        "ich fühl mi unwohl und kenn de grund nöd",
        "ich schlaf schlecht und das stresst mi",
        "ich bi dauernd erschöpft, das macht mir angst",
        "ich ha symptome, wo ich nöd einordne cha",
        "ich fühl mi nöd wie ich selber",
        "ich ha angst vor dem, was chönnt sii",
        "ich ha d ganze zit druck im chopf",
        "ich ha s gefühl, mis körpergefühl stimmt nöd",
        "ich mach mer ständig gedanke über jede kleinigkeite",
        "ich ha gärn ruhe, aber d angst bleibt",
        "ich ha schiss, dass ich öppis überseh",
        "ich weiss nöd, wie lang das no so goht",
        "ich fühl mi körperlich mega angeschlage",
        "ich ha angst, dass es nöd besser wird",
        "ich spür, dass öppis im ungleichgwicht isch",
        "ich würd gern wüsse, was genau los isch",
    ],


    ("negativ", "enttäuschung_beziehung"): [
        "ich bi grad richtig enttäuscht vo dere person",
        "ich ha s gfühl, ich bi nöd priorität für sie",
        "sie händ wieder nüd ernst gnoh, obwohl mer abgmacht händ",
        "mini erwartige sind gar nöd erfüllt worde",
        "ich ha vil investiert und wenig zrugg übercho",
        "ich fühl mi i dere beziehig nöd wahrgnuu",
        "ich weiss nöd, öb mer no am gliiche ort sind",
        "ich ha s vertroue chli verlore",
        "ich bi traurig über d entwicklung vo dem ganze",
        "ich ha s gfühl, ich bedeut nüm so vil wie früener",
        "ich würd mir meh achtig wünsche, aber es chunt nüd",
        "ich ha s gfühl, mini bemüehige werde ignoriert",
        "ich weiss nöd, wieso alles so distanziert worde isch",
        "ich bi verletzt, wie wenig rücksicht gnue wird",
        "ich ha immer wieder hoffnig, aber sie zerschlägt sich",
        "ich würd gern es gspröch führe, aber es chunt nie zäme",
        "ich fühl mi allei i dere verbindig",
        "ich weiss nöd, wieso s plötzlich so anders isch",
        "ich bi enttäuscht, dass kei initiative vo dere site chunnt",
        "ich ha s gfühl, mini gefühle werde nöd ernst gno",
        "es het nüd meh vo dem, was s mol gsi isch",
        "ich ha angst, dass s nüm besser wird",
        "ich ha s gfühl, ich kämpf allai für die beziehig",
        "ich bi müed vom ständigem probiere",
        "ich fühl mi unersetzbar und nöd wichtig",
        "ich weiss nöd, wie lang ich das no cha",
        "ich bruch eigentlich nächi, aber ich spür distanz",
        "ich würd gern vertroue, aber es fällt mer schwer",
        "ich ha s gfühl, ich bi nur option und nöd wichtig",
        "ich bi verletzt, wie wenig si zeigt, dass es ihr was bedeutet",
    ],


    ("negativ", "einsamkeit"): [
        "ich fühl mi imfall mega einsam grad",
        "au wenn lüt um mich sind, fühl ich mi allei",
        "ich ha s gfühl, niemer versteht mi würkli",
        "ich weiss nöd, mit wem ich drüber rede cha",
        "alli händ ihre gruppe, nur ich nöd",
        "ich würd mir meh nächi wünsche, aber weiss nöd wie",
        "ich ha vil kontakt online, aber wenig echt nächi",
        "ich vermisse s gfühl vo ufgnoh sii",
        "ich fühl mi scho lang isoliert",
        "ich zieh mi immer meh zrugg",
        "ich würd gern näheri verbindige ha",
        "ich ha s gfühl, ich bin unsichtbar",
        "ich fühl mi wie am rand vo allem",
        "ich weiss nöd, wie ich us däm loch usefinde",
        "ich bi oft allei mit mine gedanke",
        "ich würd gern meh soziali momänt ha",
        "ich ha s gfühl, ich rede, aber niemer loset",
        "ich fühl mi de ganze tag einsam, au unter lüte",
        "ich vermisse richtigi nächi zu öpperem",
        "ich weiss nöd, wie ich wieder anschluss finde",
        "ich ha s gfühl, ich ghör nöd richtig dezue",
        "ich würd gern es chli wärmi spüre",
        "ich fühl mi eifach leer und allei",
        "ich ha s gfühl, ich bin d ganz zit am warten uf öpper",
        "ich würd gern meh menschliche kontakt ha",
        "ich fühl mi manchmal wie verdrängt",
        "ich weiss nöd, öb öpper würkli für mich do wär",
        "ich fühl mi wie abgeschnitten vo allne",
        "ich ha angst, dass ich für immer so allein blibe",
        "ich wott eifach wieder verbindig spüre",
    ],


    # =========================================
    # POSITIV (10 Intents × 10 Bsp)
    # =========================================

    ("positiv", "dank"): [
        "merci viu mau, das isch mega lieb vo dir",
        "ich bi dir würkli dankbar für dini hilf",
        "danke, dass du dir die zit gnoh hesch",
        "es bedeutet mer vil, merci viu mau",
        "merci für dini unterstützig, das het mer guet ta",
        "ich schätz das mega, danke dir",
        "merci, dass du immer so geduldig bisch",
        "ich bin froh um dis feedback, merci",
        "danke fürs zuelose, das het mir sehr gholfe",
        "merci, dass du für mich do bisch",
        "danke für dini müeh, das isch nöd selbstverständlich",
        "ich weiss das würkli z schätze, merci",
        "merci fürs organisieren, das het perfekt passt",
        "danke dir, ich han würklich freud a dim support",
        "merci für dini ruigi art, das tuet mer guet",
        "danke für dine input, der isch mega wertvoll gsi",
        "merci, dass du mir immer wieder hilfsch",
        "ich bi dir sehr dankbar für dini perspektiv",
        "merci, dass du mir das erklärt hesch",
        "danke, dass du mi ernst nimmsch",
        "ich bi froh, dass ich cha uf dich zähle, merci",
        "merci, ich hätt das würkli nöd ohni dich packt",
        "danke, dass du dir immer wieder zit nimmsch",
        "merci fürs checke und zruggmelde",
        "danke für dini geduld mit mir",
        "merci, dass du s mir so verständlich gmacht hesch",
        "ich finds mega schön, dass du so hilfsbereit bisch",
        "danke, dass du mi unterstützt hesch, wo ichs brucht ha",
        "ich bi dir unfassbar dankbar",
        "merci, du hesch mir würklich druck gnusserlöset",
    ],


    ("positiv", "lob"): [
        "das hesch imfall mega guet gmacht",
        "ich bi voll beeindruckt vo dim resultat",
        "du hesch das richtig professionell umegsetzt",
        "das isch eifach top worde",
        "du chasch würkli stolz sii uf dich",
        "das isch super arbeit, imfall",
        "ich finds extrem gelungen",
        "du hesch en mega job gmacht",
        "ich han richtig freud ghaa, das z gseh",
        "das isch qualitativ mega stark",
        "so gueti arbeit, wow",
        "ich finds mega, wie du das aagange bisch",
        "das isch viel besser worde als erwartet",
        "du hesch voll drus, imfall",
        "ich bin begeistert vo dim output",
        "du hesch en sehr gschickte lösig gfunde",
        "ich bewundere dini sorgfalt",
        "das isch mega schön umegsetzt",
        "du hesch dä task richtig guet mastered",
        "ich find s beeindruckend, wie souverän das isch",
        "dini strukturierti art het sich voll uszahlt",
        "du hesch drüber naadacht und das merkt mer",
        "ich find s toll, wie kreativ du bisch",
        "dinu idee isch mega clever gsi",
        "ich bi positiv überrascht vo dim ansatz",
        "du hesch s thema richtig guet begriffe",
        "ich finds stark, wie du das glöst hesch",
        "de output isch sehr überzeugend",
        "ich finds mega, wie professionell du das machsch",
        "super gmachti arbeit, imfall!",
    ],


    ("positiv", "freud"): [
        "ich freu mi grad riesig drüber",
        "ich bi mega happy im momänt",
        "das macht mi richtig glücklich",
        "ich ha s gfuehl, ich strahl de ganz tag",
        "das het mir dä tag verschönert",
        "ich bi voll positive stimmig",
        "ich chan s kaum glaube, wie guet das cho isch",
        "ich bin richtig begeistert",
        "ich ha so freud gha, das z erlebe",
        "das git mir grad mega viel energie",
        "ich ha so en schöne moment gha",
        "ich bi mega erfüllt grad",
        "ich freu mi so fescht, dass das klappt het",
        "es isch eifach schön gsi",
        "ich chan grad richtig ufpüschelet atme",
        "ich bi so zufrieden mit dem ganze",
        "ich ha s so chli im herz gspürt, vor freud",
        "ich finds wunderschön, wie s cho isch",
        "ich bi mega erleichtert und froh",
        "ich han richtig freud ghaa, das z teile",
        "ich bin so glücklich über dä verlauf",
        "ich freu mi jedes mol wieder drüber",
        "ich ha das mega genosse",
        "das het mi würkli stärkt",
        "ich bin voller freud i dä situation",
        "ich han echt chli gwunder, wie guet s cho isch",
        "ich bi mega positiv überrascht worde",
        "das het so guet ta, ich ha richtig glacht",
        "ich freu mi mega über das resultat",
        "ich bi eifach glücklich grad",
    ],

    ("positiv", "erfolg"): [
        "ich ha mini prüefig bstange, bin mega erleichtert",
        "s projekt isch richtig guet usecho",
        "ich ha es ziel erreicht, wo ich lang drfür gschaffet ha",
        "mini müeh hät sich würkli glohnt",
        "ich ha es komplizierts thema endlich checkt",
        "ich ha mega positive rückmeldung übercho",
        "ich bi wiiter cho, als ich denkt ha",
        "ich bi stolz, dass ich das durchzoge ha",
        "ich ha erfolgreich es problem glöst",
        "ich ha im job en schöne erfolg gha",
        "ich spür grad richtig fortschritt",
        "ich bi mega zfriede mit mim outcome",
        "es isch besser cho als erwartet",
        "ich ha dä milestone endlich erreicht",
        "ich ha s gfühl, ich wachse grad richtig",
        "ich ha öppis gschafft, wo ich nöd sicher gsi bi",
        "ich ha feedback übercho, dass s top isch gsi",
        "ich bi stolz uf mini performance",
        "ich ha öppis erreicht, wo mir vil bedeutet",
        "ich bi mega froh über dä erfolg",
        "ich chan eifach stolz sii uf mich",
        "ich ha öppis ufgschafft, wo lang angestanden isch",
        "ich würd s am liebste allem verzelle",
        "es isch schön, öppis abghäkelt z ha",
        "ich bi imfall grad im erfolgsflow",
        "ich ha richtig en step forward gmacht",
        "ich bi so erleichtert, dass s klappt het",
        "ich ha öppis usprobiert und es het funktioniert",
        "ich bi überrascht, wie guet s usecho isch",
        "das isch sicher en schoener erfolgsmoment",
    ],


    ("positiv", "erleichterung"): [
        "ich bi so erleichtert, dass das jetzt um isch",
        "mir isch en riesen stei vom härz gfalle",
        "ich ha lang sorgä gha und jetzt isch s doch guet cho",
        "ich chan wieder freier atme",
        "mini angst het sich zum gliick nöd bestätigt",
        "ich bi froh, dass s nöd schlimmer worde isch",
        "endlich isch es chli ruiger worde",
        "ich bi so erleichtert, dass es funktioniert het",
        "es tuet mega guet, dass s jetzt klar isch",
        "ich ha lang gwartet und jetzt isch es endlich vom tisch",
        "ich bin erleichtert, dass ich nöd ganz falsch glege bi",
        "ich ha lang drüber nachdacht und jetzt passt s für mich",
        "ich bi froh, dass ich die entscheidig endlich ha",
        "ich bi erleichtert, dass ich es nöd verschlimmert ha",
        "ich chan wieder normal schlofe, so erleichtert bi ich",
        "ich bi froh, dass ich das hinter mir ha",
        "ich ha s gfühl, mini last isch chli lichter worde",
        "ich bi erleichtert, wie guet s am end usecho isch",
        "ich bi froh, dass ich nöd alles falsch verstande ha",
        "ich chan wieder es bisschen entspanne",
        "ich bi erleichtert, dass alles so friedlich verblibe isch",
        "ich bi mega froh, dass s nöd eskaliert isch",
        "endlich cha ich druck us em chopf loh",
        "ich ha s gfühl, ich ha wieder boden unter de füss",
        "ich bi erleichtert, dass ich nöd alei bin",
        "ich bi froh, dass sich die sorgä nöd bestätigt händ",
        "ich bi erleichtert, dass ich d lösig gfunde ha",
        "ich bi froh, dass ich das theama endlich abhäklet ha",
        "ich ha s gfühl, ich chan wieder duratme",
        "ich bi mega erleichtert über dä ausgang",
    ],


    ("positiv", "stolz"): [
        "ich bi würkli stolz uf mich grad",
        "ich ha öppis gstemt, wo ich mir nöd zuegtraut ha",
        "es isch schön z gseh, wie wiit ich cho bi",
        "ich ha s gfühl, ich cha zurecht stolz sii",
        "ich ha hart drfür gschaffet und s het sich glohnt",
        "ich bi zfriede mit mir und minere performance",
        "ich bi stolz, dass ich durrblibe bi",
        "ich han mini eigete erwartige übertroffe",
        "ich cha uf das resultat mit stolz luege",
        "ich bi stolz, wie ich mit dä situation umgange bi",
        "ich bi stolz, dass ich mini werte bewahrt ha",
        "ich ha mini komfortzone verliä und es het sich glohnt",
        "ich bi stolz uf jedes chliini schrittli",
        "ich bi stolz, dass ich dr schnauf ha do z'bliibe",
        "ich bi beeindruckt, wie ich das packt ha",
        "ich bi stolz, dass ich mich nöd unterkriege lah ha",
        "ich cha ehrlich säge, dass ich stolz uf mich bi",
        "ich bi stolz, dass ich s durchzieh ha",
        "ich bi stolz uf mini entwicklung",
        "ich cha mini erfolg würkli spüre",
        "ich bi stolz, dass ich so vill gelernt ha",
        "ich bin stolz, dass ich Verantwortung übernoh ha",
        "ich bi stolz, dass ich mini gränze akzeptiere cha",
        "ich bi stolz, wie ich aus fehler glärnt ha",
        "ich bi stolz uf mini disziplin",
        "ich bi stolz, wie ich das gemeistert ha",
        "ich ha endlich s gfühl, ich cha öppis",
        "ich bi stolz, dass ich mi nöd ha stressä lah",
        "ich bi stolz, dass ich mini ziele verfolge",
        "ich bi stolz uf mini schritt i d richtige richtung",
    ],


    ("positiv", "verbindung_freunde"): [
        "mir händ en mega gmüetliche abig gha",
        "ich bi so froh um mini fründschaftä",
        "mir händ so viel glacht, das het guet ta",
        "ich fühl mi grad richtig ufgnoh i dere gruppe",
        "ich ha s gfühl, mir verstöhnd üs immer besser",
        "mir händ sogar über ernsti sache chönne rede",
        "ich schätz es mega, so mensche um mich z ha",
        "ich ha lang nüm so e ehrlechs gspröch gha",
        "ich bin dankbar für die nächi, wo mir händ",
        "ich ha s gfühl, ich bi nöd alei mit mine themä",
        "mir händ en schöne momänt zäme gha",
        "ich bi froh, dass ich so verlässligi fründe ha",
        "d zeit mit mine fründe tuet mer unglaublich guet",
        "es isch schön, dass mer eifach sii cha wie mer isch",
        "ich ha mich richtig wohl gfühlt bi üse abmachig",
        "ich ha s gfühl, mini fründe gäh mir mega viel halt",
        "ich bi dankbar für jede minute zäme",
        "ich ha fründ, wo ehrlich zu mir sind – das isch gold wert",
        "ich finds schön, dass mir so offe chönd rede",
        "mini fründe gäh mir richtig viel energie",
        "ich ha wieder mol gmärkt, wie wichtig fründe sind",
        "ich bi mega froh, so lüt i mim läbe z ha",
        "ich fühl mi unterstützt und nöd alei",
        "mir händ so en schöne vibe zäme",
        "ich fands wunderschön, wie authentisch s gsi isch",
        "mini fründe sind würkli wertvoll für mi",
        "ich bi richtig dankbar für üs allne",
        "ich ha s gfühl, mini fründschaftä wärded immer tiefer",
        "ich ha en mega schöne momänt mit öpper erlebt",
        "es isch so guet, sich verstanden z fühle",
    ],


    ("positiv", "motivation"): [
        "ich bi imfall voll motiviert grad",
        "ich ha richtig lust, öppis azpacke",
        "ich bi voller energie im momänt",
        "ich ha s gfühl, jetzt isch en guete moment für neues",
        "ich ha so vill idee, die ich umsetze wott",
        "ich spür grad richtig drive i mir",
        "ich bi mega parat für die nächschte schritt",
        "ich ha sogar freud am challenge",
        "ich fühl mi richtig produktiv",
        "ich ha s gfühl, ich cha grad mega vill bewege",
        "ich bin motiviert wie scho lang nüm",
        "ich schmöck grad es chli erfolgs luft",
        "ich bi voll im fokus",
        "ich ha mega bock zum dranzbliibe",
        "ich glaub, ich cha richtig gueti ideä umsetze",
        "ich bi voll im flow grad",
        "ich würd am liebschte grad starte",
        "ich spür mega viel kraft im momänt",
        "ich bin motiviert, öppis neu z probiere",
        "ich ha s gfühl, alles isch möglich grad",
        "ich ha voll lust uf s projekt",
        "ich bi mega driven",
        "ich glaub, es chunt öppis guets use",
        "ich bin richtig wach und parat",
        "ich ha richtig en schub übercho",
        "ich bi mega inspiriert",
        "ich ha viel energie für dini ufgabe",
        "ich bi voller elan",
        "ich spür en positivie schub",
        "ich ha wieder richtig lust zum öppis erreiche",
    ],


    ("positiv", "vorfreude"): [
        "ich freu mi mega uf d feriä",
        "ich ha so vorfreud uf dä termin",
        "ich chan s kaum erwarte, bis es so wiit isch",
        "ich bi scho am plane und ha mega freud",
        "ich ha e guets gfühl für das, was chunt",
        "ich zähl scho d tägli bis zu dem tag",
        "ich freu mi uf d lüt, wo ich wieder gseh",
        "ich ha lang druf gwartet, endlich passierts",
        "ich ha s gfühl, das wird öppis mega schöns",
        "mini vorfreud git mir richtig energie",
        "ich bi richtig hibbelig vor excitement",
        "ich freu mi ufglich, dass bald öppis neus asteht",
        "ich chan s kaum nochem nöd abwarte",
        "ich ha richtig bock uf das, was chunt",
        "ich ha s gfühl, das wird mega guet",
        "ich ha mega vorfreud uf mini plän",
        "ich freu mi uf die nächschti wuche wie wild",
        "ich bi parat, das chli abentüür z starte",
        "ich freu mi uf öppis, wo mir vil bedeutet",
        "ich han lang usgharrt, jetzt isch s bald wiit",
        "ich spür richtig chli kribbele vor freud",
        "ich chan nüm still hocke vor excitement",
        "ich freu mi uf jede minute i dä zukunftige phase",
        "ich ha mega freud am drübernachdenke, was chunt",
        "ich gseh dem tag richtig entgegen",
        "ich ha so freud im chopf, wenn ich dra denke",
        "ich bi richtig hyped uf das",
        "ich han mega vorfreud, ehrlich",
        "ich ha s gfühl, das wird en guete moment",
        "ich chan etz scho nüm abschalte vor freud",
    ],


    ("positiv", "zufriedenheit_alltag"): [
        "ich bi imfall recht zfriede mit mim alltag aktuell",
        "es lauft nöd perfekt, aber es stimmt für mich",
        "ich ha e gueti balans zwüsche arbeit und pause",
        "ich ha s gfühl, s passt grad würkli",
        "ich cha guet lebe mit dem tempo im momänt",
        "mini rutine tuet mir richtig guet",
        "ich bi froh, dass kei drama isch",
        "ich ha s gfühl, ich ha mis läbe im griff",
        "ich cha sogar chli ruhe gniesse",
        "ich bi dankbar für die stabilität, wo ich grad ha",
        "ich bin entspannt und zufrieden",
        "ich ha imfall en ruhigi phase, wo mir guet ta",
        "ich ha das gfühl, ich bin genau im richtige flow",
        "ich bi voll im einklang mit mim alltag",
        "ich chan mich sogar über chliini sache freue",
        "ich fühl mi grounded und ruhig",
        "ich finds schön, wie ausgegliiche alles isch",
        "ich ha en angenehmi routine, wo passt",
        "ich bi dankbar, dass s im momänt so harmonisch isch",
        "ich bin nöd überforderet, nöd gelangweilt – genau richtig",
        "ich ha s gfühl, alles läuft im gesunde rahme",
        "ich bi froh um jede ruigi minute",
        "ich chan mich mit guetem gfühl zruglehne",
        "ich bi zfriede mit mim alltagstempo",
        "ich ha s gfühl, ich stimm richtig ab mit mir selber",
        "ich finds schön, dass ich nöd hetze mues",
        "ich bi stabil und entspannt",
        "ich chan s läbe grad chli gniesse",
        "ich bin glücklich mit mim rhytmus",
        "ich ha s gfühl, ich bi grad am richtige ort im läbe",
    ],


    # =========================================
    # NEUTRAL (10 Intents × 10 Bsp)
    # =========================================

    ("neutral", "smalltalk"): [
        "hey, wie gahts dir hüt?",
        "hoi zäme, was machsch grad so?",
        "na, alles klar bi dir?",
        "wie isch dini wuche bisher gsi?",
        "bisch guet id wuche startet?",
        "was lauft so bi dir im momänt?",
        "lange nüm gseh, wie gohts?",
        "was treibsch aktuell so?",
        "wie hesch s gester gha?",
        "alles guet bi dir, so grundsätzlich?",
        "was hesch für plän hüt?",
        "wie steigsch i dä tag?",
        "hesch öppis cools vorhe?",
        "wie goht s ganz allgemein?",
        "was hesch bis jetzt vom tag gha?",
        "hüt schöns wetter, oder?",
        "wie gaht s dir im job aktuell?",
        "wie hesch s am wucheend gha?",
        "alles im grüene bereich bei dir?",
        "wie lauft s bi dir privat?",
        "was hesch gester so agstellt?",
        "wie goht s dim umfeld so?",
        "hesch öppis spannends erlebt letztens?",
        "wie isch dini stimmig hüt?",
        "wie lauft s im momänt so im allgemeine?",
        "alles ruhig bi dir oder chli stressig?",
        "wie isch dä tag bisher so gsi?",
        "wie hesch du de morn aagfange?",
        "na, wie treibsch dich so dur dä tag?",
        "wie geits vo 1 bis 10 hüt?",
    ],


    ("neutral", "orga"): [
        "wänn wür dir passe zum treffe?",
        "chömer s eventuell chli verschiebe?",
        "passt dir morn oder eher am abig?",
        "wottsch lieber online oder vor ort abmache?",
        "welche uhrziit wär für dich am praktischste?",
        "wie lang hesch du ungefäär zit?",
        "söll ich dir die details schicke?",
        "wottsch en kalender-eintrag?",
        "für wänn sömer s ungefähr apegge?",
        "chan s au spöter in de wuche sii, falls nötig?",
        "is für dich eher früeh oder spoot besser?",
        "wo chömer üs am einfachste treffe?",
        "wöllmer s fix mache oder no offen lah?",
        "chan ich dir zwei, drei vorschläg mache?",
        "wie schnell bruchsch die info?",
        "is für dich okay, wenn mer s chli flexibel lah?",
        "söll ich dr no en reminder setze?",
        "wottsch dass ich di abhol?",
        "is die zeitfenster realistisch für dich?",
        "söll ich dr no öppis vorbereite?",
        "wie wottsch s organisiera – kurz oder ausführlich?",
        "is s äntweder oder? oder beides möglich?",
        "sömer grad zwei date fixe mache?",
        "chan ich dir no alternative uhrzeite schicke?",
        "isch eher diese wuche oder nöchschti besser?",
        "willmer s spontan lah und morn no mal luege?",
        "wo wär für dich am angenehmste ort?",
        "chan ich dir öppis koordiniera?",
        "söll ich no lüt ahschriebe?",
        "sömer das via chat oder call abgleiche?",
    ],


    ("neutral", "frage_info"): [
        "wie funktionieret das genau?",
        "chasch mir kurz erkläre, was ich mues mache?",
        "was meinsch, isch die bescht option?",
        "wie lauft das normalerwiis ab?",
        "was isch dä unterschied zwüsche däne zwei variante?",
        "gits öppis, wo ich im vorus sött wüsse?",
        "mues ich dafür öppis speziells vorbereite?",
        "chan ich das nacher no ändere, falls nötig?",
        "hesch erfahrung mit dem gha?",
        "wie lang gaht das ungefäär?",
        "was bruch ich für die ufgabe?",
        "was wär dä erste schritt?",
        "wofür isch das eigentlich da?",
        "brauchts dafür es account oder nöd?",
        "chan ich das usprobiere, bevor ich mich entscheide?",
        "mues ich auf öppis bestimmts luege?",
        "welche variante würdisch du empfehle?",
        "was isch dä normalfall bi dem prozess?",
        "gits öppis, wo mer gerne verwechslet?",
        "isch das kompliziert oder eher easy?",
        "was passiert, wenn ich en schritt falsch mache?",
        "chan ich das spöter no korrigiere?",
        "mues ich dafür support ahfrage?",
        "gits öppis, wo mer gern übersiht?",
        "wie würdest du das mache?",
        "was sind vor- und nachteile vo dem?",
        "welches tool bruucht mer am beschte?",
        "isch es relevant, wie ich s formuliere?",
        "brucht s en bestimmte reihenfolg?",
        "wie würdest du s am eifachste erkläre?",
    ],

    ("neutral", "sonstiges"): [
        "isch mir imfall relativ egal",
        "mer chönd s au eifach offen lah",
        "mer lueged, was sich ergit",
        "ich han no kei klare meinig dezue",
        "mer chönd s spontan entscheide",
        "es muess nöd unbedingt hüt sii",
        "ich chan mit beidem lebe",
        "so wichtig isch s imfall nöd",
        "mer chönd s no chli beobachte",
        "ich find s weder mega guet no mega schlecht",
        "ich bi da ziemlich flexibel",
        "ich cha mich dem easy ahpasse",
        "es isch nöd tragisch, wie mer s mache",
        "ich han da kei priorität",
        "es spielt kei so roll für mich",
        "mer chönd s drum au verschiebe, falls nötig",
        "ich bi da ganz entspannt",
        "mir isch s ehrlich gseit egal",
        "ich ha da kei starke preference",
        "es muess nöd grad jetzt gfillt werde",
        "mir chönd s liässig halte",
        "ich cha damit guet umgah, egal wie mer s mache",
        "ich find, mer chönd s locker ahgah",
        "isch nöd es thema, wo mi stresst",
        "mer chönd s denn sponti entscheide",
        "ich bin nöd so fest investiert i das",
        "beidi optione sind okay für mich",
        "ich bi da offen für alles",
        "chum, mer mache s eifach so, wie s grad passt",
        "ich glaub, das chunt scho guet, wie mer s mache",
    ],


    ("neutral", "status_update"): [
        "kurzes update: es lauft imfall solid, aber langsam",
        "ich bi grad mitten im process",
        "es isch no nöd alles fertig, aber ich bi dra",
        "ich ha s gröbschte gmacht, aber es brucht no feinarbeit",
        "ich bi no nöd ganz dort, wo ich wott sii",
        "bis jetzt isch no nüt eskaliert, alles chillig",
        "ich ha es bitzeli fortschritt gmacht",
        "im momänt isch s chli viel, aber im rahme",
        "es lauft, aber ich bruch no es chli zit",
        "ich gib dir bschaid, sobald öppis neus isch",
        "es chunnt langsam, aber sicher",
        "ich bi druf und dra, langsam vorwärts",
        "nöd spektakulär, aber es bewegt sich öppis",
        "ich bi imfall steady am schaffe",
        "es isch no weni fertig, aber s chunnt guet",
        "ich mues no es paar sache optimiere",
        "ich gib mir müeh, Schritt für Schritt",
        "ich bi imfall no dran, kei sorge",
        "update: es funktioniert zu 80% aktuell",
        "es bruucht eifach no chli geduld",
        "ich bi grad am usprobieren vo verschidene sache",
        "ich mues no öppis teste, bevor ich sicher bi",
        "ich bi am dokumentiere, aber es dauert",
        "es isch guet unterwegs",
        "ich mach morgä wiiter, hüt nüm",
        "es het sich nöd viel verändert, aber ich bi dran",
        "ich bi eifach am durchackere",
        "ich ha dr status chli aktualisiert",
        "ich bi zwüschet dur chli im stress, aber okay",
        "ich meld mich, sobald ich e klarheit ha",
    ],


    ("neutral", "planung_langfristig"): [
        "ich versueche mini längfristige plän z ordne",
        "ich überleg mer, wie s nöchschte jahr sött usegseh",
        "ich ha nöd ganz klar, wohi, aber ich wott struktur iibringe",
        "ich probier imfall chli vorus z plane",
        "ich möcht mir es paar ziel setze",
        "ich bin am luege, wie sich job und privat verbinde lah",
        "ich wott meh ruhe i mini plän ine bringe",
        "ich mues no ussortiere, was mir würkli wichtig isch",
        "ich denk grad vil drüber nach, wie s langfristig witergoht",
        "ich probier Schritt für Schritt öppis z verändere",
        "ich ha s ziel, mini ressourcen besser z nutze",
        "ich möcht chli planklarheit schaffe",
        "ich bin am reflektieren über nächste schritt",
        "langfristig wott ich es paar sache ändere",
        "ich möcht luege, was nachhaltig passt",
        "ich probier grad s grosse ganze z gseh",
        "ich ha s gfühl, ich bruch en neustart i ein paar bereiche",
        "ich wott guet vorbereitet sii uf s, was chunt",
        "ich probiere, ziele realistischer z setze",
        "ich ha s gfühl, ich wachs chli in öpperes neus ine",
        "ich plane langsam, aber durchdacht",
        "ich ha e vision, aber no nöd alles detaili",
        "ich wott längerfristig stabilität ufbaue",
        "ich probier bewusster z entscheide",
        "ich reflektier grad vil über mini zukunft",
        "ich wott d priorität neu setze",
        "ich probier chli strategie uftue",
        "ich möcht langfristig meh ruum für mich ha",
        "ich plane drüber, wie ich wott schritt für schritt wiitercho",
        "ich probiere, d richtung richtig z fühle",
    ],


    ("neutral", "feedback_sachlich"): [
        "nur chli feedback: es isch übersichtlich, aber chönnt no klarer sii",
        "grundsätzlich guet, aber es het no verbesserigspotenzial",
        "vo minere sicht funktioniert s, aber d doku chönnt besser sii",
        "ich find s sachlich okay, struktur passt",
        "ich würd no meh erläuterige erwarte",
        "d zeitplän chönnt optimiert wärde",
        "inhaltlich imfall solid, optisch chönd mer no schaffe",
        "es isch lösbar, aber d anleitig isch knapp",
        "ich find s fair, aber chli streng formuliert",
        "vo mir us es nüchternes aber positives feedback",
        "es isch guet, aber nöd ganz konsistent",
        "ich würd no meh beispiel integriere",
        "d gliederig isch guet, aber chli steif",
        "ich würd drüber luege, öb mer s flüssiger mache cha",
        "ich finds sachlich, aber wenig lebhaft",
        "mini rückmeldung: chli meh details wär guet",
        "es isch solid, aber chli trocken",
        "ich ha s gfühl, mer chönt s no visuell unterstütze",
        "es isch verständlich, aber lang",
        "ich würd die abschnitt klarer trenne",
        "es het struktur, aber chli chaotisch",
        "ich finds eigentlich guet, aber chli technisch",
        "ich würd en mini-redesign empfählen",
        "es hed substanzi, aber chli repetitiv",
        "mini empfehlig: meh praxisbezug",
        "ich würd eso la, aber optimierbar isch s immer",
        "es isch logisch, aber wenig intuitiv",
        "es würd nüt schade, wenn mer s chli kürzt",
        "ich ha s feedback nüchtern analysiert",
        "es isch guet, aber nöd wow",
    ],

    ("neutral", "hobby_talk"): [
        "ich bi grad wieder vil am gamä",
        "ich go gern jogge, wenn ich kopf lüfte möcht",
        "ich ha wieder aagfange meh z läse",
        "ich bi voll im kochfilm im momänt",
        "ich lueg grad vil series zum abchille",
        "musik lose hilft mer zum entspanne",
        "ich bi am übe uf mim instrument",
        "ich ha neu mit fotografie aagfange",
        "ich bi gern drausse, wenn s wetter stimmt",
        "ich probier grad meh kreativität i mis läbe z bringe",
        "ich mache gern chli bastleprojekt",
        "ich ha wieder meh lust uf games",
        "ich ha en neue workout-routine",
        "ich versueche neue rezepte us",
        "ich bi imfall gern am stricke oder häkle",
        "ich lese grad en mega spannende romanreihe",
        "ich bi viel am zeichne in letzter zit",
        "ich ha wieder zu skateboard gfunde",
        "ich verbring gern zit i de natur",
        "ich cha stundenlang podcasts lose",
        "ich lern grad öppis neus im bereich design",
        "ich probier grad en neue sportart us",
        "ich bi imfall am gärtnere, mega chillig",
        "ich interessier mi grad für astronomie",
        "ich sammle chli vinyl-platte",
        "ich bi gern am improvisiere bim musigiere",
        "ich entdecke grad mini alte hobbys wieder",
        "ich bi am experimentiere mit digital art",
        "ich schätz ruigi hobbys wie puzzlen",
        "ich ha gern alles, wo kreativität bruucht",
    ],


    ("neutral", "news_teilen"): [
        "hesch gseh, was grad i de nöi passiert isch?",
        "ich ha es spannends artikel glese über das thema",
        "es het wieder es update zu dere sach geh",
        "imfall, d app het jetzt es neus feature",
        "ich ha ghört, dass sich öppis a de regel gänderet het",
        "es lauft grad en spannende entwicklung i dem bereich",
        "ich bi über en beitrag gstolpert, wo guet passt",
        "grad i de nöi git s vil diskussion drum",
        "es git en neus projekt, wo spannend chönt sii",
        "ich ha gseh, dass d termi chli gschobe worde sind",
        "ich ha öppis interessantes uf social media gseh",
        "es het en bericht geh, wo ich mega spannend gfunde ha",
        "es git es update zur situation, falls di das interessiert",
        "ich ha grad es video gseh, wo guet erklärt, was lauft",
        "es het en neue entwurf für die regelige geh",
        "ich ha grad en newsletter über das thema becho",
        "es het öppis neus im technologiebereich",
        "ich bi über en artikel gstolpert, wo relevant chönnt sii",
        "ich ha gseh, dass en event churz vor verschobig gstande isch",
        "es het neue info, aber no nöd 100% bestätigt",
        "ich ha gsehen, dass s projekt offiziell startet",
        "ich bi über en trend gstolpert, wo grad ufchunnt",
        "ich ha neus glese über öppis, wo mir im chopf blibe isch",
        "es het en update zu dere debatte geh",
        "ich ha gseh, dass öppis i de statistik ufgfüert worde isch",
        "es git neue erkenntnisse zu em thema",
        "ich ha en bericht gseh über es zukunftsprojekt",
        "ich ha gseh, dass öppis in planig isch",
        "es het en neue entwiklig, aber no früeh",
        "ich ha s gfuehl, es chunnt da bald öppis gross us dem use",
    ],


    ("neutral", "technik_support"): [
        "mini app spinnt scho wieder und ich weiss nöd warum",
        "ich chum nöd druus mit dem update",
        "mis wlan macht grad komische sache",
        "ich weiss nöd, wie ich das programm installiere",
        "dä login will eifach nöd funktioniere",
        "ich ha s gfühl, mis handy isch mega langsam worde",
        "ich bechum ständig fehlermeldig und check nöd, was s heisst",
        "ich weiss nöd, welches kabel wohi ghört",
        "ich ha öppis verstellt und find s nüm",
        "ich bruch chli hilf, zum das wieder zum laufe bringe",
        "mis gerät reagiert mega verzögert",
        "ich weiss nöd, wie ich d einstellige zruggsetze",
        "mis programm stürzt die ganze zit ab",
        "ich ha kei plan, wieso der button nüt macht",
        "ich weiss nöd, ob ich öppis falsch installiert ha",
        "ich find die funktion nüm im menü",
        "ich ha s gfühl, mini verbindig bricht ständig ab",
        "ich weiss nöd, wie ich das update rückgängig mache",
        "mis gerät zeigt komische icons a",
        "ich cha eifach nöd druf zugriffe",
        "ich weiss nöd, wie ich das dokument exportiere",
        "ich weiss nöd, wieso d app nöd synchronisiert",
        "ich cha d datei nüm ufmache",
        "ich ha fehler im system, aber kei ahnig, woher",
        "ich weiss nöd, wie mer das backup macht",
        "mini lautsprecher funktioniere plötzlich nüm",
        "ich cha s passwort nöd zruggsetze",
        "ich weiss nöd, wie ich d verbindig wiederherstelle",
        "ich ha abstürz und kei ahnig, wieso",
        "ich bruch chli technischa support, ehrlich",
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
        "Ich verstand, dass das grad schwierig für dich isch. Was belastet di im Momänt am meiste?",
    ("negativ", "hilfe_bitten"):
        "Ich probier gern z helfe. Sag eifach, wo du grad hängsch.",
    ("negativ", "stress_alltag"):
        "Alltag cha schnell z vil wärde. Womit fangemer am beschte aa, zum s chli z ordne?",
    ("negativ", "konflikt"):
        "Konflikt sind nöd eifach. Was belastet di am meiste i dere situation?",
    ("negativ", "selbstzweifel"):
        "Es isch okay, wenn mer sich mal unsicher fühlt. Worin hesch du grad am meiste zweifel?",
    ("negativ", "überforderung_job"):
        "Jobstress cha würklich belaste. Was druckt dich im momänt am meiste?",
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
        "Hoi! Schön vo dir z ghöre. Was lauft so bi dir?",
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
        "Klingt spannend. Wenn du wotsch, chasch gern no meh drüber verzelle.",
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
# 3) Dataset-Build-Funktionen (ohne Augmentation)
# =========================================================

def build_base_dataset(
    out_csv: str = DATA_CSV_BASE,
) -> pd.DataFrame:
    """Seed-Basisdatensatz bauen (nur EXAMPLES, keine Augmentation)."""
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

    # doppelte Kombinationen aus Text/Label/Intent entfernen (zur Sicherheit)
    base_df = base_df.drop_duplicates(
        subset=["text", "label", "intent"]
    ).reset_index(drop=True)

    # Preprocessing für Modell/Features
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
    # Nur Datensätze aus den EXAMPLES erstellen (ohne Augmentation)
    build_base_dataset()
    build_chatpairs_dataset()

