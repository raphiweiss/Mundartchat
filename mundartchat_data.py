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
    # Verneinung / Normalisierung von "nöd"
    # "nöd" -> "noed" (durch Umlaut-Normalisierung), hier Vereinheitlichung auf "ned"
    "noed": "ned",

    # Formen von "sein"
    # "bin" kommt häufig vor, "bi" ist deine Zielsform
    "bin": "bi",

    # Formen von "können"
    # du verwendest oft "cha" / "chan" – wir ziehen alles auf "chan"
    "cha": "chan",

    # kommen / gehen
    # du hast sowohl "chunt" wie "chunnt" im Dialekt üblich, Vereinheitlichung
    "chunt": "chunnt",
    # "gang"/"geh" -> "go", damit nur eine Grundform
    "gang": "go",
    "geh": "go",

    # Ausdrücke
    # "imfall" verwendest du sehr häufig – hier leicht normalisiert
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
        "so chli langsam mag ich nüm, es isch eifach nur müehsam",
        "kei ahnig wieso, aber s läuft dähei immer irgendöppis quer",
        "es het wieder mol nüt so funktioniert, wie mer s vorgstellt het",
        "langsam chunnt mer s gfühl, die sache wird nie richtig",
        "jedä schritt isch komplizierter als er müessti sii",
        "ehrlech, das isch wieder mol typisch chaoseinsatz",
        "wersch wahnsinnig bi dem gschluder, ehrlich",
        "ständig öppis anders, aber selten öppis guets",
        "gopf, das cha doch nöd sii, so schwer isch s doch nöd",
        "ha grad nüme nerv für so halbbatzigi lösige",
        "das het di qualität vo enere kartoffel, würkli",
        "es isch eifach nume zum ufpasse und chnurre",
        "jedä versprochene schritt verzögert sich, wie immer",
        "mini hoffnig schmilzt jedes mol direkt wieder zäme",
        "so unorganisiert hani s lang nüm erlebt",
        "es läuft drunter und drüber, und niemer reagiert",
        "kann doch nöd sii, dass mer immer no am gliiche punkt schteit",
        "grad wenn mer denkt, etz chunnt ruhe, wird s wieder schlimmer",
        "es regt eine so unglaublich uf, so öppis",
        "lüüt, es isch würkli nöd normal, wie schlecht das gmacht isch",
        "machts doch eifach mol richtig, isch das z viel verlangt?",
        "so viel chaose uf einisch – nid mal lustig",
        "wär hät das plant? s wirkt improvisiert bis zum abwinke",
        "jesses gott, jedes mol s gliiche theater",
        "s isch langsam würkli ermüdend, alles zieht sich wie kaugummi",
        "mer het s gfühl, jede lösig macht no e neus problem uf",
        "ha scho vil gseh, aber das da isch next level unorganisiert",
        "es isch zum d chrämpf chaufe, so ineffizient",
        "wär zum geier hät die idee gha? es funktioniert ja gar nöd",
        "immer öppis, aber selten öppis guets – langsam reichts",
    ],



    ("negativ", "frust"): [
        "grad ehrlich, ich mag nüm – es isch z viel uf einisch",
        "fühlt sich a, als würd mi alles überhole",
        "momentan chunt mer grad nüt richtig us de händ",
        "s ganze druckt mer uf s gemüt, extrem",
        "ich häng nur no irgendwie in dä tag ine",
        "wärsch froh, ich chönt e chli abstand ha",
        "mini energie isch komplett im keller",
        "grad jede chliini sache bringt mi durenand",
        "ha s gfühl, ich laufe im blindflug ume",
        "nid emal meh d einfachste sache gäh mer grad vo de hand",
        "s isch alles so schwergängig im momänt",
        "chan mi eifach nöd ufraffä, egal was ich probier",
        "alles brucht doppelt so vil kraft wie normal",
        "im chopf isch s chaos, weiss nöd emal wo aafange",
        "wär schön, mol es licht am end vom tunnel z gseh",
        "fühl mi wie abgeschnitte vo minere eigete energie",
        "ha s gfühl, ich verlür grad komplett dr faden",
        "irgendwie funktioniert nüt, weder drinnä no dussä",
        "würd so gern e pause druf ha, aber s lebe laht nöd zue",
        "jedä versuech strengt mi übertriebe muetig a",
        "grad alles chli z viel für mis närevgerüst",
        "ich such dr drive – aber er chunnt eifach nöd zrugg",
        "ha s gfühl, ich bin im dauerstruggle mode",
        "alles, was eigentlich easy wär, fühlt sich schwer a",
        "chan mi grad selber nöd richtig usstoh",
        "irgendwie steck ich voll im leerlauf",
        "wär schön, chönnt ich mol druck ablo",
        "ich kämpf ume jeden millimeter energie",
        "fühlt sich a, als wär ich innerlich am limit",
        "es macht mi extrem mürb, echt",
    ],



    ("negativ", "hilfe_bitten"): [
        "chum grad überhaupt nöd drus – chasch mol e blick druf werfe?",
        "wär mega, wenn mer das kurz chönntisch erläuterä",
        "ha s probiert, aber s macht kei sinn, ehrlech",
        "chan mer das Schritt für Schritt dürgo?",
        "irgendwie checki dä teil nöd ganz",
        "hesch e minn, mir das chli usenand z näh?",
        "mues es gseh, glaub – chasch s mir vormache?",
        "sit e wuche kämpfi mit dem, chasch mir helfe?",
        "danke, wenn du mir da chli unterstützig gäh chasch",
        "ha so vil frage, weiss gar nöd, wo aafange",
        "chöntisch du drüberluege und säge, öb das stimmt?",
        "versteh s prinzip, aber nöd, wie mer s umsetzt",
        "kannsch mir bitte es praktisches beispiel gäh?",
        "ha s gfühl, ich überseh öppis Wichtigs",
        "irgendwie hängt s – ke ahnig, was ich falsch mach",
        "wär froh um en tipp, wie mer das am beschte aapackt",
        "chasch das eifach chli vereinfachä für mich?",
        "s tönt logisch, aber ich bring s nöd i d praxis",
        "bi grad komplett verwirrt, chasch helfe?",
        "mues das versteh, aber i bi no nöd ganz debii",
        "ha dä Schritt hundertmol gläse, dennoch nöd verstanden",
        "gibsch mir chli guidance, wie s würkli gmeint isch?",
        "chan mer s zäme kurz duregah – glaub, ich mach öppis falsch",
        "brauch dringend e erklärig, sonnst gang ich under",
        "irgendwie chli überforderet da – chasch mir ushelfe?",
        "wär super, wenn du s mir anders chönntisch formulieren",
        "komm grad nöd uf dä knackpunkt, chasch mir en hinwiis gäh?",
        "ha s gmacht, aber s funktioniert nöd – was überseh ich?",
        "bin nid sicher, öb das so stimmt – chasch drüberluege?",
        "kapier s ehrlich nöd – chasch mir en gefallen mache und s kurz erklärä?",
    ],



    ("negativ", "stress_alltag"): [
        "grad alles uf einisch, chumi nüm drus mit dem tempo",
        "hetzt vo termin zu termin, kei sekunde zum verschnuuufe",
        "fühlt sich a, als wär dä tag z churz für alles",
        "immer öppis los, niemert git mir ruhe",
        "ha nonig mol richtig aagfange, und scho stah ich im stress",
        "s ufstaple wird immer grösser, egal was ich abhake",
        "bin grad nume am umehetze, ohne pause",
        "i de morgä scho im roten bereich, richtig müehsam",
        "ha kei luft meh zwüschet dä ganzä sache",
        "jedä tag fangt im stress aa und hört so uf",
        "d agenda platzt fasch, nöd normal",
        "es brucht nüm viel, und ich chäm völlig under",
        "ha s gfühl, ich renn dr wuche nur no hinterhere",
        "kopf isch voll, herz isch voll, aber d zeit isch leer",
        "cha mich uf nüme konzentriere vor lauter druck",
        "alles geit so schnäll, ich chum kaum na",
        "wär schön, mol e momänt ohni dauerstress z ha",
        "wie s würd ein dauernd wechsle zwüsche hundert sache",
        "bin dür schnuf – wortwörtlich",
        "ha kei puffer meh für spontane ufwand",
        "jedä tag bringt meh ufgabä als ich abbring",
        "fühl mich, als würd ich im autopilot laufe",
        "es isch nume no mache, mache, mache… ohni reset",
        "bin erschöpft, bevor dä tag überhaupt richtig startet",
        "brucht dringen es chli entspannig, sonscht gang ich d wände ufe",
        "alles so verdichtet, kei freiraum meh",
        "wär froh, nur mol ein tag ohni planig und to-dos",
        "min chopf isch e riesige to-do-lischte worde",
        "lauter ufdrag, aber kei platz zum verschnuufe",
        "würdi gern chli entschleunige, aber s laht mi grad nöd",
    ],



    ("negativ", "konflikt"): [
        "schon wieder sind mir im gschpräch total ane vorbei",
        "s eskaliert so schnäll, ich chas kaum glaubä",
        "het grad wieder mol im streit ändet, obwohl s gar nöd hätt misse",
        "jedä versuech, sachlich z rede, lauft in d hose",
        "es isch, als reded mer zwei komplett verschideni sprachä",
        "immer wenn ich öppis erläuterä wott, wird s verdreht",
        "fühl mi nöd im mindescht ernst gno oder gseh",
        "d atmosphär isch dermass ufglade, mer chunt nöd zu lösige",
        "mer triggered üs grad gegenseitig viel z schnäll",
        "jedä kleine hinweis wird grad gross gmacht",
        "ha s gfühl, mir händ kei gemeinsame basis meh",
        "es tuet weh, aber mer stahnd ir kommuniktion völlig im sturm",
        "statt lösige z sueche, schiebt mer sich nur d schuld zue",
        "irgendwie reded mer umeinander ume und nüt chunt a",
        "chumm mir nüm druus, was dä ander eigentlich wott säge",
        "mer händ s nie lang ruhig, sofort gibts wieder reibige",
        "mini energie isch am bodä – jede diskussion laugt mi uus",
        "ha langsam angst, dass mer üs immer wiiter entferned",
        "es isch so müehsam, wenn nüt im gspröch stabil bliibt",
        "hab s gfühl, ich muss immer nachgeh, sonscht chunt gar nüt",
        "wär schön, mol en dialog ohni unterschwelligi aggression",
        "doch jedes mol, wenn ich öppis klarstelle wott, gibts krach",
        "irgendwie lauft s jede mol i d gleiche destruktivi dynamik",
        "niemer loset wirkli zue, mer wartet nume uf s gegä-argument",
        "so chömer nöd wiiter – d spannig isch zum schneide dick",
        "bi grad verzweiflet, wie mer das je wider chönd flickä",
        "jede kleine uneinigkeid brennt grad wie benzin im feuwer",
        "statt nächer cho, drifted mer immer wiiter usenand",
        "weiss nöd, wohi das führt, aber s macht mi fertig",
        "ich such nur nach ruhe, aber jedä dialog bricht zäme",
    ],


    ("negativ", "selbstzweifel"): [
        "mer kommt s vor, als würdi nöd ganz genüege",
        "so oft stell ich mi selber infrage, es strengt mi a",
        "im momänt fällt s mir schwer, a mini fähigkeit z glaube",
        "ha manchmal s gfühl, alli andere chönd s besser",
        "mini unsicherheit nimmt grad chli überhand",
        "jedä fehler löst grad vil z viel grübelei us",
        "weiss nöd, öb ich würkli so stabil bin, wie ich gern wär",
        "bin schnell verunsichert, wenn öppis nöd perfekt lauft",
        "ha s gfühl, mini leistig stimmt nöd so ganz",
        "lueg oft uf mich selber z streng zrugg",
        "irgendwie fällt s mir schwer, stolz uf mis mache z sii",
        "mini erwartige an mich selber sind viel z hoch",
        "mues viel zu oft bestätigung sueche, sonscht wacklets",
        "ha müeh, mini stimmig nöd vo zweifel überflüggä z lah",
        "so kleine sache brenged mi manchmal komplett us em tritt",
        "frag mi oft, öb ich würkli am richtige ort bi",
        "würdi gern meh ruhigheit i mini eigete bewärtig ha",
        "mer verwirrt mich, dass ich trotz müeh so unsicher bliibe",
        "fühl mi nöd ganz sicher i dem, was ich grad mach",
        "mini innere kritiker isch grad chli laut",
        "es chliini kompliment würd mir scho guet tue, glaub",
        "mer fählt dr gloubä, dass ich s würkli cha",
        "immer im vergleich mit andere – müehsam",
        "ha s gfühl, ich stah mir selber chli im weg",
        "bin so schnell verletzlich, wenn öppis nöd klappt",
        "würd gern lernä, milder mit mir umzgah",
        "weiss nöd, öb ich dä schritt würkli packe cha",
        "mer macht druck uf sich, und denn blockierts",
        "ha s gfühl, min selbstwert wacklet grad chli",
        "möcht gern wieder meh vertroue i mi selber ha",
    ],



    ("negativ", "überforderung_job"): [
        "i de arbeit staut sich grad alles, chumi kaum no hindernaa",
        "d ufwänd im büro wachsed grad schnäller als ich türe cha",
        "het so vil parallel laufend, verlür fast dr Überblick",
        "immer meh task, aber d zeit bliibt di gliiche",
        "ha s gfühl, s workload drückt mer langsam uf s gemüt",
        "bin scho am morgen überlade mit anfragä",
        "es lauft so hektisch, chumi nüme richtig i e ruhephase",
        "min arbeitsstapel wird höher statt chliner",
        "würd gärn konzentriert schaffe, aber s chunnt ständig öppis dazwüsche",
        "fühl mi vo termine und deadlines zämmegschobä",
        "d koordination vo allem isch grad e riesigs gschluder",
        "mues überall gleichzeitig sii und das strengt a",
        "ha s gfühl, mini kapazität isch grad voll ufgfüllt",
        "s git kei freiraum zum durchatme untertag",
        "d erwatige sind momentan extrem hoch",
        "mer rührt so vil zue, weiss nöd, wo aafange",
        "cha nach em feierabend nüme richtig abschalte",
        "ständigi umpriorisierung macht mich total dureenand",
        "s büro isch grad wie en dauerwetschlauf",
        "würd dringend e chli stabilität im job bruuche",
        "bi grad im dauerbetrieb und das merkt mer langsam",
        "jedä tag bringt meh ufwänd als ich abarbeite cha",
        "d termindruck nimmt mer grad vil energie",
        "ha s gfühl, ich schaffe nume no im reaktonsmodus",
        "mer fehlt es team, wo chönnt chli druck abfange",
        "d mengen an emails isch nöd normal im momänt",
        "ha so vil offeni sache, dass ich mi selber zwenig priorisiere",
        "z vil verantwortig uf einisch, es überfahrt mi chli",
        "wär froh, es gäb e tag mit weniger druck",
        "weiss nöd, wie lang das pensum no so cha wiitergah, isch grad richtig streng",
    ],



    ("negativ", "gesundheit_sorge"): [
        "irgendwie macht mir mini körpergefühl grad chli sorge",
        "fühl mi seit tage komisch und weiss nöd recht, warum",
        "ha s gfühl, dä körper sendet mir chli seltsami signäl",
        "symptome chömed und gönd, und das verunsichert mi",
        "immer wieder gedanke: hoffentlich isch nüt ernsthaftes",
        "bin unsicher, öb ich mol soll go abkläre",
        "d müedigkeit isch untypisch für mich, macht mer chli angst",
        "fühl mi körperlich nöd stabil im momänt",
        "im ganzen körper isch so e unruh, nöd ganz klar woher",
        "mach mir vil z vil gedanke über das alles",
        "mis imunsystem fühlt sich chli wacklig a",
        "irgendwo mues en grund sii, aber ich find nüt",
        "bin nöd sicher, ob s villicht schlimmer chönnt werde",
        "weiss ehrlich nöd, wie ich das einordne söll",
        "fühl mi unwohl, aber chas nöd genau beschriebe",
        "schlaf unruhig und das nagt an mir",
        "d ständige erschöpfig irritiert mi",
        "het symptome, wo ich nöd richtig zueordne cha",
        "fühl mi eifach nöd ganz wie ich selber",
        "ha respekt vor em, was s chönnt sii – will aber nüt dramatisiere",
        "druck im chopf, immer wiederkehrend",
        "mis körpergefühl isch eifach nöd im gleichgewicht",
        "mues uf jede kleinigkeid aachte, wils mich verunsichert",
        "wär froh, ich könnt chli beruhiger d situation bschaue",
        "habe angst, dass ich öppis überseh, wo wichtig wär",
        "es irritiert mich, dass s scho so lang ahaltet",
        "fühl mich körperlich chli angeschlage in letzter zit",
        "frag mi, ob s villicht nur stress isch oder öppis anders",
        "spür, dass nüt ganz im lot isch, aber weiss nöd was",
        "wett eigentlich nume wüsse, was grad los isch mit mir",
    ],



    ("negativ", "enttäuschung_beziehung"): [
        "mer tuet s weh, wie distanziert das grad worde isch",
        "hätt echt erwartet, dass si chli meh zeigt, dass es ihr wichtig isch",
        "mini bemühige scheined im leere z verpuffe",
        "ha s gfühl, d prioritäte lieged nüm bi üs",
        "si het s abgmacht wieder nöd ernst gnoh",
        "es stimmt eifach nöd, wie wenig rückmeldung chunnt",
        "ha vil investiert, aber nöd viel zrugg gspürt",
        "d distanz zwüsche üs macht mi traurig",
        "irgendwie passt s nüm, aber weiss nöd recht warum",
        "ha s gfühl, es het nüm di wärmi wie früener",
        "würd mir meh achtig und klarheit wünsche",
        "mini wort chömed nöd würkli a bi ihr, so wirkt s",
        "ha s vertroue chli aakratzt",
        "es irritiert mich, wie wenig initiative vo ihrer site chunnt",
        "i de verbindig fühl ich mi grad chli allei",
        "s isch enttäuschend, wie schnell alles abglitte isch",
        "ha s gfühl, mini gefühle werde nöd so wahrgnuu",
        "weiss nöd, ob mir überhaupt no uf em gliiche level sind",
        "d distanz macht s schwierig, eifach normal z rede",
        "mini erwartige sind leider nöd erfüllt worde",
        "es het nüt meh vo dem vertroue, wo s mol gits het",
        "han mir meh nächi erhofft, aber s passiert wenig",
        "gseh viel müeh vo minere site, aber wenig zrugg",
        "bin immer wieder enttäuscht, wie wenig reaktion chunnt",
        "würdi gern es offes gspröch ha, aber es chunt nie so wiit",
        "irgendwie stimmt d balance nüm – alles fühlt sich einseitig a",
        "ha s gfühl, ich bi chli zrugggstellt i ihrem läbe",
        "brucht eigentlich bestätigung, aber es chunnt fast nüt",
        "es tüend weh, dass es so wenig echo git uf mis bemüeh",
        "frag mi, wie mir wieder nächer chönd cho, so wie s mol gsi isch",
    ],



    ("negativ", "einsamkeit"): [
        "fühlt sich im momänt alles chli vereinzelt a",
        "au mitten unter lüte chunnt kei richtigi nächi uf",
        "mer chunt s vor, als wär ich gedanklich abgschotte",
        "weiss nöd, mit wem ich das chönnt teile",
        "alli scheined ihr umfeld z ha, während ich suchä",
        "wär froh um meh nächi, aber weiss nöd, wo aafange",
        "viel onlinekontakt, aber wenig, wo sich richtig ahfüelet",
        "vermisse e gfüehl vo ufgnoh sii",
        "schon längi chli isoliert am fühle",
        "zieh mi in letzter zit eher zrugg",
        "würd gern verbindigere momänt erlebe",
        "manchmal fühlt s sich a, als wär ich nöd ganz sichtbar",
        "ha s gfühl, ich stah chli am rand vo gruppä",
        "such scho lang wieder es plätzli, wo ich dezueghörä",
        "bi vil allei mit mine gedanke unterwägs",
        "wär schön, öppert zum spontane austausch z ha",
        "rede scho, aber s chund wenig zrugg",
        "tag dur tag e ähnlechs gfüehl vo distanz",
        "seh anderi lähnig zueene, und mir fehlt s chli",
        "weiss nöd grad, wie ich wieder meh sozialität finde",
        "ha s gfühl, ich bi nöd ganz teil vo dem, was lauft",
        "würdi gärn wärmi und nächi spüre i kontakt",
        "e gfüehl vo leere begleitet mi chli",
        "wart oft uf momänt, wo s chönnt chli näher werde",
        "seh oft mensche, aber d verbindig fehlt",
        "manchmal fühlt s sich a wie e kleine unsichtbari wand drumume",
        "bin chli verunsichert, wer würkli für mich do wär",
        "s gfühl vo abgränzig isch immer wieder da",
        "frag mi, wie ich Schritt für Schritt wieder meh nächi chönnt ufbaue",
        "wünsch mir e verbindig, wo sich echt und tragend aafühlt",
    ],



    # =========================================
    # POSITIV (10 Intents × 10 Bsp)
    # =========================================

    ("positiv", "dank"): [
        "merci viu mau, das het mi würkli freut",
        "huere dankbar für dini hilf, ehrlich",
        "wär mega lieb vo dir gsi – merci!",
        "dis engagement bedeutet mir echt vil, danke dir",
        "merci für dini unterstützig, het genau s bruucht",
        "ich schätz dini müeh enorm, merci viu mau",
        "so geduldig wie du bisch – danke dir herzlich",
        "merci für dis feedback, het mi total witerbrocht",
        "danke fürs zuelose, das het mir richtig guet ta",
        "so schön, dass du für mich do bisch – merci!",
        "danke für dini zeit, nöd selbstverständlich",
        "ich bin dir ehrlich wahnsinnig dankbar",
        "merci fürs organisiere, isch perfekt cho",
        "dini ruigi art het mer mega gholfe – danke!",
        "merci für dine wertvolle input",
        "danke, dass du immer wieder drus looksch",
        "ich find s mega schön, wie hilfsbereit du bisch – merci",
        "danke, dass du s mir so klar erklärt hesch",
        "bin froh, chan ich uf dich zähle – merci viu mau",
        "merci fürs drüberluege und zruggschriebe",
        "danke für dini geduld, het mer enorm guet ta",
        "danke dir, ich hätt s ohni dich nöd so hinchriegt",
        "merci, dass du dir immer wieder zit nimmsch",
        "dis feedback het richtig sitte gäh – danke!",
        "ich find s riesig, wie du mi unterstützt hesch – merci",
        "danke, dass du mich ernst nimmsch und zuelose tuesch",
        "merci, dine support het mi voll entlastet",
        "es isch nöd selbstverständlich, was du machsch – danke dir",
        "fühl mi richtig supported vo dir – merci!",
        "danke für alles, wirklich!",
    ],



    ("positiv", "lob"): [
        "wow, das isch imfall richtig stark worde",
        "bin mega beeindruckt, wie du das aagange bisch",
        "dä output isch grad huere professionell",
        "so gueti arbeit – cha nüt anders säge",
        "channsch stolz sii uf das resultat",
        "dä einsatz het sich voll uszahlt",
        "richtig gelungeni umsetzig, bravo!",
        "en mega job, ehrlich",
        "het mich richtig gfreet, das so z gseh",
        "qualitativ sehr stark, kompliment",
        "wärkli top niveu, super gmacht",
        "find s hammer, wie souverän du bisch bliibe",
        "viel besser worde als erwartet, gratuliere",
        "dä vibe vo dim projekt isch mega positiv",
        "bin begeistert vo dim klare ansatz",
        "dinu lösig isch huere gschickt und kreativ",
        "seh voll, wie sorgfälting du bisch vorgange",
        "wärklich schön umegsetzt bis ins detail",
        "dä task hesch du locker gmeistert",
        "beeindruckend, wie ruhig und strukturiert du blibsch",
        "dini überlegige simmer richtig guet gsi",
        "find s toll, wie ideenreich du schaffsch",
        "dinu input isch clever und sehr präzis gsi",
        "bin positiv überrascht vom resultat",
        "du hesch s thema mega guet durschaue",
        "starker output – merkt mer, dass viel gedanke dri steckt",
        "dä approach isch echt überzeugend",
        "professionell bis zum schluss, tiptop",
        "huere schön glöst, bravo!",
        "wirkli super arbeit – grosses kompliment!",
    ],


    ("positiv", "freud"): [
        "das het mi grad richtig glücklich gmacht",
        "wouh, so viel freud uf einisch!",
        "bin imfall mega happy im momänt",
        "das gfühl vo freud isch grad enorm schön",
        "dä tag isch mir grad viel heller worde",
        "so en schönes erlebnis, ehrlich",
        "chans fasch nöd glaube, wie guet das cho isch",
        "bin voll begeischtert vo dem ganze",
        "die situation het mir mega energie geh",
        "han so richtig chli im herz gspürt, vor freud",
        "het mich total ufpüschelet",
        "eifach nur schön, wie das usecho isch",
        "bin grad voll im freudeschub",
        "het mi überrascht, wie positiv das verlafe isch",
        "wärmigs gfühl, das mi de ganz tag begleited",
        "d freud isch grad überragend gross",
        "das het mir richtig guet ta – han grad glacht",
        "bin so erleichtert und glücklich zäme",
        "das resultat het mi extrem gfroot",
        "ha das erlebnis mega gnosse",
        "jedes mol, wenn ich drüber nachdenke, chumts mir e lächle",
        "fühl mi grad mega lebendig und fröhlich",
        "chan s nöd abstreite: ich strahl grad chli",
        "s isch so e befreiig, dass s so guet cho isch",
        "die freud isch hütt scho mehfach ufblitzt",
        "bin mega positiv überrascht worde",
        "echt en moment, wo mir extrem guet ta het",
        "han richigi freud daran gha, das z teile",
        "fühlt sich eifach wunderbar a",
        "bin voll erfüllt vo dem schöne gfühl",
    ],


    ("positiv", "erfolg"): [
        "mini prüefig isch bstange – riesigi erleichterig!",
        "dä projektabschluss isch imfall mega guet worde",
        "en langfrischtig ziel erreicht, wo ich ewig drfür gschaffet ha",
        "mini müeh het sich definitiv uszahlt",
        "endlich es komplizierts thema komplett checkt",
        "hei, mega feedback übercho – het mich richtig gfroot",
        "bin weiter cho, als ich eigentlich denkt ha",
        "stolz, dass ich s wirklich durchzieht ha",
        "es schwierigs problem erfolgreich glöst",
        "im job en richtig schöne erfolg gha",
        "spür grad klaren fortschritt, Schritt für Schritt",
        "mega zfriede mit em ganze outcome",
        "viel besser cho als ich erwartet ha",
        "dä milestone isch endlich abghäklet",
        "fühl mi voll im wachstum i dem bereich",
        "öppis glöst, wo ich erst nöd sicher bi gsi, ob s gaht",
        "top rückmeldung becho – het richtig guet ta",
        "bin stolz uf mini ganz performance",
        "en lieblingsziel erreicht, wo mir vill bedeutet het",
        "so froh über dä erfolgsmoment",
        "darf würkli e chli stolz sii uf mich",
        "e ufgab erledigt, wo lang uf de liste gstande isch",
        "würd s am liebschte grad mit alli teile",
        "mega guets gfühl, öppis richtig abgschlosse z ha",
        "bin imfall grad voll im flow",
        "huere step forward gmacht – spürs richtig",
        "so erleichteret gsi, wie sauber das klappt het",
        "es usprobiert und es isch grad perfekt ufgange",
        "bin selber überrascht, wie guet s usecho isch",
        "definitiv en richtig schöner erfolgsmoment",
    ],



    ("positiv", "erleichterung"): [
        "endlich – das het mir grad richtig druck gnau",
        "mer isch sprichwörtlich en stei vom härz gfalle",
        "ha lang sorgä gha, drum tuet s doppelt guet, dass s jetzt stimmt",
        "chann wieder entspannt atme, so erleichtert",
        "zum gliick het sich mini sorge nöd bestätigt",
        "so froh, dass s nöd schlimmer worde isch",
        "endlich isch wieder ruhe i d situation cho",
        "huere guets gfühl, dass s am end funktioniert het",
        "het richtig entspannt, dass d sache jetzt klar sind",
        "so lang druf gwartet – und jetz isch s vom tisch, mega",
        "bin erleichtert, dass mini einschätzig nöd falsch gsi isch",
        "ha viel drüber grüeblet, aber jetzt passt s für mich",
        "d entscheidig het guet ta, fühl mi leichter",
        "erleichteret gseh, dass ich nüt verschlimmert ha",
        "ha wieder ruiger nacht gha – es tuet guet",
        "so schön, das chli hinter mir z wüsse",
        "mini lascht fühlt sich definitiv lichter a",
        "froh, wie guet s am schluss usecho isch",
        "schön z merke, dass ich s nöd komplett falsch verstande ha",
        "het richtig entspannt, wieder chli ruhe z ha",
        "froh, dass s friedlich und ohne drama bliibe isch",
        "erleichtert, dass s nöd i falschi richtung gange isch",
        "dä druck im chopf isch endlich wäg",
        "ha s gfühl, ich ha wieder stabilität under de füess",
        "truurig wär s gsi, aber bin froh, dass ich nöd alei bim themä bi",
        "bin erleichtert, dass sich mini sorgä nöd bestätigt händ",
        "mega happy über d lösig, wo ich gfunde ha",
        "endlich chani s theama abhäkeln und loslah",
        "han s gfühl, ich chan wieder normal schnufe",
        "d erlöserig über dä positive usgang isch riesig",
    ],



    ("positiv", "stolz"): [
        "het richtig guet ta, z merke, wie stolz ich chan sii",
        "ha öppis gstemt, wo ich mir lang nöd zuegtraut ha",
        "schön z gseh, wie wiit ich i dere zit cho bi",
        "fühlt sich richtig an, stolz uf mis leiste z sii",
        "ha hart drfür gschaffet – und s het sich voll glohnt",
        "bi ehrlich zfriede mit minere performance",
        "bin stolz, dass ich drblibe bi, au wenn s streng gsi isch",
        "han mini eigete erwartige sogar übertroffe",
        "uf das resultat chan ich würkli mit stolz luege",
        "stolz, wie ruhig und überlegt ich reagiert ha",
        "huere schön, dass ich mini werte bewahrt ha",
        "mini komfortzone verlasse und trotzdem erfolg gha – mega!",
        "cha stolz sii uf jedes chliini schrittli",
        "finds echt cool, dass ich dr schnauf gha ha zum drbliibe",
        "bi vo mir selber beidrukt, wie ich das packt ha",
        "stolz drauf, dass ich mich nöd ha verunsichere lah",
        "cha ohne scheu säge: ich bi stolz uf mich",
        "bin stolz, dass ich s würkli durchzieh ha",
        "merks richtig: mini entwicklung goht vorwärts",
        "der erfolg het mir e chli schub geh",
        "bin stolz, wie viel ich gelernt ha i dere phase",
        "stolz, dass ich verantwortig übernoh ha und guet damit umgange bi",
        "cool z merke, dass ich mini gränze akzeptiere cha",
        "freu mi, wie vill ich us frühere fehler mitgnoh ha",
        "mini disziplin het sich würkli uszahlt – stolz!",
        "es gfühl vo stolz, wie ich das gmeistert ha",
        "ha endlich s gfühl, ich cha öppis bewirke",
        "finds cool, dass ich mi nöd ha stressä lah",
        "mini ziele verfolge und drbi bliibe – das macht mi stolz",
        "füühlt sich guet a, dass ich i d richtige richtung underwägs bi",
    ],



    ("positiv", "verbindung_freunde"): [
        "dä abig isch imfall mega gmüetlich gsi, het richtig guet ta",
        "so froh, so ehrligi fründschaft im läbe z ha",
        "mir händ so viel glacht – pure energie",
        "het sich mega schön aagfühlt, wie ufgnoh ich mi i dere gruppe gfühlt ha",
        "mer merkt richtig, wie mir üs immer besser verstöhnd",
        "het guet ta, au über ernsti sache chönne rede",
        "ich schätz so mensche extrem, wo ehrlich und warm sind",
        "lang nüm so es offes und ehrlechs gspröch gha",
        "für die nächi unter fründe bin ich mega dankbar",
        "so schöns gfühl, nöd alei mit mine themä z sii",
        "dä gemeinsame moment isch würkli schön gsi",
        "verlässligi fründe wie ihr sind gold wert",
        "d gemeinsami zit tuet mir jeweils extrem guet",
        "isch schön, eifach völlig sich sälber chöne sii dra",
        "ha mi richtig wohl gfühlt bi üsem treffe",
        "mini fründ gäh mir richtig viel halt im alltag",
        "jede minute zäme isch wertvoll für mich",
        "fründ wo ehrlich blibed, sind für mich mega wichtig",
        "finds schön, wie offe und natürlich mir chönd rede",
        "mini fründe gäh mir grad voll energie",
        "han wieder mol gmerkt, wie wichtig so fründschaft isch",
        "fühl mi mega getragen vo üsem umfeld",
        "dä vibe unter üs isch einfach angenehm und herzlich",
        "ha s treffe richtig wunderschön gfunde",
        "mini fründe bedeuted mir extrem viel",
        "bin dankbar für üsi gmeinschaft als team",
        "würd sach, mini fründschaftä wärded vo jahr z jahr tiefer",
        "het mir mega vill bedeutet, dä momänt z teile",
        "so guet, sich verstande und akzeptiert z wüsse",
        "fühlt sich schön a, so supportive fründe z ha",
    ],



    ("positiv", "motivation"): [
        "grad voll dä drive i mir – würd am liebschte sofort loslegä",
        "so richtig bock, öppis aazpacke im momänt",
        "energie pur – ich spürs richtig",
        "fühlt sich a, als wär jetzt dä perfekte moment für öppis neus",
        "es sprudlet nur so vo idee, möcht grad alles umsetze",
        "dä innere schwung isch mega präsent",
        "parat für die nächschte schritt – fühlt sich guet a",
        "seh sogar freud i de kommende challenge",
        "bin grad überraschend produktiv unterwegs",
        "ha s gfühl, ich cha grad güet bewege mit minere energie",
        "motiviert wie scho lang nüm – richtig angenehm",
        "schmöck förmlich erfolgsluft i de nöchi",
        "bin voll konzentriert und im fokusmodus",
        "dranzbliibe fällt mir grad mega eifach",
        "spür, dass gueti ideä parat stöhnd",
        "bin voll im flow – jede schritt fäut sich natürlich a",
        "wär am liebschte grad direkt gstertet",
        "kraft und elan sind grad ufem maximum",
        "hä richtig lust, öppis neus uuszprobe",
        "es gfühl, dass grad mega vil möglich isch",
        "freu mi richtig uf s projekt, es kribbelt scho",
        "bin mega driven und parat zum uusrisse",
        "han im gfühl, es chunt öppis sehr positiivs use",
        "fühl mi wach, klar und startbereit",
        "han richtig en motivationsschub übercho",
        "bin mega inspiriert vo dem, was chönnt werde",
        "energie für die ufgabe isch definitiv da",
        "voller elan – chönnt grad e stund durcharbe",
        "d vibes sind mega positiv, tut richtig guet",
        "han wieder en richtig starke wille, öppis z erreiche",
    ],



    ("positiv", "vorfreude"): [
        "d vorfreud isch grad riesig – chan s kaum erwarte",
        "freu mi huere uf d feriä, wird sicher mega",
        "ha so freud uf dä termin, chönt jetzt scho sii",
        "bin scho voll am plane und chum richtig i stimmig",
        "ha e super gfühl für das, was bald chunt",
        "zelle fascht jede tag, bis dä moment endlech do isch",
        "freu mi enorm uf d lüt, wo ich wieder gseh",
        "ha lang druf gwartet – jetzt passiert s gli!",
        "han s gfühl, das wird öppis richtig schöns",
        "mini vorfreud git mir voll energie",
        "bi richtig hibbelig, jedes mol wenn ich dra denke",
        "freu mi mega, dass bald öppis neus asteht",
        "chan s nüme so lang abwarte, ehrlich",
        "voll bock uf das, was als nöchsts chunt",
        "es gfühl säit mir, dass das mega guet wird",
        "mini plän für di nöchschti zit mache mi voll glücklich",
        "würkli gspannt uf die nächschti wuche",
        "parat fürs abentüür, wo vor mir staht",
        "freu mi uf öppis, wo mir richtig vil bedeutet",
        "ha lang ufgfahre, und jetzt isch s bald sowiit",
        "spür richtig s kribbele vo vorfreud",
        "chan fasch nüm still hocke – so excited",
        "freu mi uf jede chliini minute vo dem, was chunt",
        "ha mega freud, scho nur bim drübernohdenke",
        "gseh dem tag richtig fröhlich entgegen",
        "jedes mol wenn ich dra denke, chunnt en smile",
        "mached mich mega hyped, as chunt echt guets",
        "ha grad mega viel vorfreud uf das ganz projekt",
        "merks voll: das wird en richtig schöner moment",
        "vorfreud isch so gross, chani fasch nüm abschalte",
    ],



    ("positiv", "zufriedenheit_alltag"): [
        "grad e richtig angenehmi ruä im alltag – tuet mega guet",
        "nöd perfekt, aber s stimmt für mich und das reicht völlig",
        "d balans zwüsche arbeit und pause passt grad super",
        "ha s gfühl, alles füegt sich im momänt guet zäme",
        "s tempo isch angenehm – chan guet damit läbe",
        "mini tägliche routine git mir voll stabilität",
        "schön, mol kei drama umezha – richtig entspannt",
        "fühl mi grad stabil und im griff mit allem",
        "chan mer sogar öppis ruä gönne – selten aber schön",
        "bin mega dankbar für die stabilität, wo grad da isch",
        "entspannt und zfriede – gueti kombi",
        "froh um die ruhig phase, het richtig guet ta",
        "s fühlt sich grad wie en natürlecha flow a",
        "bin voll im einklang mit mim alltag",
        "chan mich sogar über chliini sache freue – das macht vill uus",
        "grounded und ruhig, eifach angenehm",
        "es isch schön, wie ausgegliiche alles wirkt",
        "mini routine isch schlicht, aber passt perfekt",
        "bin sehr dankbar, dass s grad harmonisch lauft",
        "nöd überforderet, nöd gelangweilt – genau dä sweet spot",
        "ha s gfühl, alles bewegt sich i enem gsunde rahme",
        "jede ruigi minute isch wie es chli gschenk",
        "chan mich mit ruigem gfühl zruglehne",
        "s alltags-tempo stimmt – weder z schnell noch z langsam",
        "fühl mich im einklang mit mir selber",
        "schön, dass ich grad nöd hetze mues",
        "stabil, entspannt und gut unterwegs",
        "chan s läbe grad eifach chli gniesse",
        "bin sehr zfriede mit mim rhythmus",
        "ha s gfühl, ich bi grad am richtige ort im läbe",
    ],



    # =========================================
    # NEUTRAL (10 Intents × 10 Bsp)
    # =========================================

    ("neutral", "smalltalk"): [
        "hey, wie lauft s so bi dir hüt?",
        "hoi zäme, alles locker im momänt?",
        "na, wie goht s dir grad?",
        "wie isch dini wuche bis jetzt so cho?",
        "bisch guet i de tag gstarrt?",
        "was treibsch aktuell so ume?",
        "lange nüm gseh – wie gaht s?",
        "was machsch imfall grad so?",
        "wie hesch s gister so gha?",
        "alles okay bi dir, grundsätzlich?",
        "hesch öppis vor hüt oder eher entspannt?",
        "wie steigsch so i dä morgä?",
        "isch öppis spannends aagleit für dich?",
        "wie gaht s dir im grosse und ganze?",
        "wie het dä tag bisher usgseh bi dir?",
        "schöns wetter hüt – hesch s gniesse chönne?",
        "wie lauft s jobmässig grad?",
        "wie isch s wucheend so gsi bi dir?",
        "isch alles im grüene bereich?",
        "wie lauft s grad privat so?",
        "was hesch gister so angestellt?",
        "wie goht s dim umfeld, alles easy?",
        "hesch letztens öppis cools erlebt?",
        "wie isch dini stimmig hüt unterwegs?",
        "wie würdsch dini situation grad beschriebe?",
        "isch s grad eher ruhig oder meh stressig bi dir?",
        "wie het sich dä tag bis jetzt aagfühlt?",
        "wie hesch du dä morn aagfange?",
        "na, wie rennsch so dur dä tag hüt?",
        "wenn du müesst – wie wär dini tageslaunä vo 1 bis 10?",
    ],


    ("neutral", "orga"): [
        "wänn wür dir am beschte passe fürs treffe?",
        "chömer s eventuell chli verschiebe, falls nötig?",
        "passt dir morn, oder isch am abig besser?",
        "wottsch lieber online abmache oder persönlich?",
        "welchi uhrziit chämer anpeile bi dir?",
        "wie viel zit hesch ungefäär zur verfüegig?",
        "söll ich dir d details grad schnell schicke?",
        "wottsch, dass ich dir en kalender-eintrag mach?",
        "für wänn söttemer s grob apegge?",
        "chönnts au spöter i de wuche passe bi dir?",
        "isch früeh oder eher spoot angenehmer für dich?",
        "wo wär am praktischste ort für s treffe?",
        "wötsch s fix mache oder lieber no chli flexibel lah?",
        "chan ich dir paar vorschläg mache, was würke chönt?",
        "wie dringend bruchsch die info?",
        "isch okay, wenn mer s vorerst locker plane?",
        "söll ich dir no en kleine reminder setze?",
        "wottsch, dass ich di unterwegs grad abhole?",
        "passt das zeitfenster für dich oder isch s z knapp?",
        "söll ich im vorus no öppis vorbereite?",
        "wie wottsch s lieber – kurz und unkompliziert oder detailliert?",
        "ischs äntweder-oder oder beidi optione möglich bi dir?",
        "wöllmer grad zwei date fix mache, falls eins nöd passt?",
        "chan ich dir alternativ uhrzeite schicke?",
        "isch diese wuche realistischer oder eher nöchschti?",
        "wöllmer s spontan lah und morn no mal abgleiche?",
        "wo wär für dich am angenehmste ort fürs treffen?",
        "söll ich dini koordinatig überneh und lüt ahschriebe?",
        "passt es dir, wenn ich no paar sache organisiere?",
        "wöllmer das kurz via chat oder lieber mitem call abgleiche?",
    ],



    ("neutral", "frage_info"): [
        "wie funktionieret das eigentlich im detail?",
        "chasch mir kurz erkläre, was ich genau mues mache?",
        "was würdisch als bescht option aaluege?",
        "wie lauft dä prozess normalerwiis ab?",
        "was isch dä unterschied zwüsche däne zwöi varianten?",
        "gits öppis, wo ich vorhär unbedingt sött wüsse?",
        "mues ich dafür irgendöppis speziells vorbereite?",
        "chan ich das spöter no ändere, falls s nötig isch?",
        "hesch selber erfahrung mit dem gha?",
        "wie lang nimmt das ungefähr i anspruch?",
        "was bruch ich alles, um das korrekt z mache?",
        "was wär dä logische erste schritt?",
        "für was isch das eigentlich genau gmacht?",
        "brauchts dafür en account oder gaht s ohni?",
        "chan ich das testweise usprobiere, bevor ich mich entscheide?",
        "mues ich bi öppis bestimmtem speziell ufpasse?",
        "welchi variante würdisch du spontan empfehle?",
        "wie isch dä gewöhnliche ablauf bi dem?",
        "gits sache, wo mer schnell verwechslet bi dem thema?",
        "isch das kompliziert oder würdisch säge eher easy?",
        "was passiert, wenn ich dä schritt falsch mach?",
        "chan ich fähler im nachhinein korrigiere?",
        "mues ich dafür eventuell support ahhole?",
        "gits typischerweise öppis, wo mer gern übersiht?",
        "wie würdisch du persönlich das aagoh?",
        "was sind so die vor- und nachteili vo dä lösig?",
        "welches tool bruucht mer ide regel am effizienteste?",
        "isch relevant, wie genau ich das formuliere?",
        "mues ich en bestimmte reihenfolg iihalte?",
        "wie chasch mir das am eifachste i zwei, drü sätz erkläre?",
    ],


    ("neutral", "sonstiges"): [
        "isch mir imfall relativ egal, wie mer s mache",
        "mer chönd s gern eifach offen lah und luege",
        "mir lueged mal, was sich draus ergit",
        "han no kei klare meinig dezue, ehrlich",
        "chömer s gern spontan entscheide",
        "es muess definitiv nöd hüt sii",
        "für mich sind beidi wege absolut okay",
        "so wichtig isch s imfall würkli nöd",
        "mer chönd s eifach no chli beobachte",
        "finds weder besonders guet noch schlecht",
        "bi da eigentlich recht flexibel",
        "chan mich dem ganz easy ahpasse",
        "es spielt kei riesigi roll, wie mer s umegsetze",
        "han da ehrlich kei priorität",
        "für mich isch s chillig, wie mer s mache",
        "mer chönd s au verschiebe, falls s nöd passt",
        "bi da ganz entspannt und offen",
        "mir isch s grundsätzlich egal, würkli",
        "han kei stärk preference i die riichtung",
        "es muess nöd grad jetzt erledigt werde",
        "mer chönd s locker u meg eifach halte",
        "ich cha mit de meiste optione guet läbe",
        "find, mer sött s nöd überkompliziere",
        "isch nöd es thema, wo mich stresst",
        "wenn s dir lieber isch, chömer s spontan lösä",
        "bin nöd wahnsinnig investiert i die frag",
        "beidi varianten tönd absolut okay für mich",
        "bi da völlig offen, ehrlich",
        "mer mache s eifach so, wie s grad am bestä passt",
        "glaub, das chunt scho guet, egal wie mer s löse",
    ],



    ("neutral", "status_update"): [
        "kurzes update: es lauft imfall solid, aber langsam",
        "bin grad mitten im process und mach Schritt für Schritt",
        "no nöd alles fertig, aber ich bi dr blibe",
        "s gröbschte isch done, jetzt bruuchts no feinarbeit",
        "bin no nöd ganz am ziel, aber nah drann",
        "bis jetzt isch alles stabil, nüt eskaliert",
        "es het sich chli öppis beweegt, langsam aber sicher",
        "im momänt chömed viel sache zäme, aber isch im rahme",
        "es lauft, aber ich bruuch no chli zit für dä rest",
        "gib dir bschaid, sobald öppis neus passiert",
        "progress isch da, aber eher im slow-motion-style",
        "bin steady am schaffe, es chunnt Schritt für Schritt",
        "nöd mega spannend grad, aber es geht vorwärts",
        "update: ich bi no voll im arbeitsmodus",
        "es isch no weni fertig, aber s zeichnet sich ab",
        "mues no paar dinge usbügle",
        "ich arbeite dran, Schrittli für Schrittli",
        "bin imfall no dra, kei stress",
        "status: würd säge ca. 80% funktionieren aktuell",
        "bruucht halt eifach no chli geduld",
        "bin grad am teste vo verschidene variante",
        "mues no sicherheitsschritt mache, bevor ich’s final mach",
        "bin am dokumentiere – das nimmt chli zit",
        "es isch guet unterwegs und macht sinn so",
        "ich mach morn wiiter, hüt isch gnue gsi",
        "het sich nöd viel verändere, aber mini liste schrumpft",
        "bin grad am durchackere, aber alles ok",
        "habe grad de status aktualisiert – chli Fortschritt isch da",
        "zwüschet dur chli stress, aber grundsätzlich ok",
        "meld mi wieder, sobald ich klarheit ha",
    ],



    ("neutral", "planung_langfristig"): [
        "bin grad drann, mini langfristige plän chli z ordne",
        "überleg mir, wie s nöchschte jahr ungefähr sött usegseh",
        "ha no nöd alles klar, aber ich wott meh struktur iibringe",
        "probier imfall chli vorus z plane statt nur spontan",
        "möcht mir paar realischtischi ziel setze fürs nöchschti",
        "lueg grad, wie sich job und privat besser verbinde lah",
        "wott meh ruhe und stabilität i mini plän ine bringe",
        "mues no ussortiere, was mir langfristig würkli wichtig isch",
        "denk grad vil drüber nach, wohi s über Zyt wott go",
        "ändere Schritt für Schritt paar sache, aber nüt überstürzt",
        "möcht mini ressourcen langfristig gscheiter nutze",
        "probier e chli klarheit i mis grössers bild z bringe",
        "bin am reflexiere über min nöchschti weg",
        "langfristig wott ich es paar gewohnheite ändere",
        "möcht luege, was nachhaltig zu mir passt",
        "versueche s grössere ganze besser z überblicke",
        "ha s gfühl, ich bruche chli neui orientierig",
        "wott guet vorbereitet sii uf s, was chönnt chöme",
        "möcht mini ziele realistischer und weniger stressig setze",
        "ha s gfühl, ich wachse langsam i öppis neus ine",
        "plane imfall eher langsam, aber überlegt",
        "ha es visionli im chopf, aber no nöd alles usformuliert",
        "wott langfristig meh stabilität ufbaue",
        "probier bewusster und nöd automatisch z entscheide",
        "reflektiere im momänt vil über mini zukunftsrichtung",
        "möcht mini priorität chli neu sortiere",
        "lüge grad, weli strategie längerfristig gscheit wär",
        "wott langfristig meh ruum für mich selber ha",
        "plane chli, wie ich Schritt für Schritt wiitercho wott",
        "probier d richtung z spüre, bevor ich grössers aareiss",
    ],



    ("neutral", "feedback_sachlich"): [
        "chli feedback: es isch übersichtlich, aber no nöd ganz klar strukturiert",
        "grundsätzlich guet umegsetzt, aber es het sicher optimierungspotenzial",
        "vo minere sicht her funktioniert s, aber d dokumentation chönnt präziser sii",
        "ich find s sachlich okay, d struktur stimmt soweit",
        "würd eventuell no meh erläuterige a paar stelle ergänze",
        "d zeitplän chönd mer chli straffer formuliere",
        "inhaltlich solid, optisch chönd mer s sicher nachschärfe",
        "es isch lösbar, aber d anleitig isch eher knapp",
        "find s fair formuliert, aber chli streng im ton",
        "vo mir us en nüchterne, aber positive rückmeldung",
        "es isch guet gmacht, aber nöd ganz konsistent i de detail",
        "würd empfehle, no e paar beispiel dezue z nehme",
        "d gliederig isch logisch, aber wirkt chli steif",
        "würd mall ueluege, öb mer s chli flüssiger mache cha",
        "sachlich i ordnig, aber wenig dynamisch",
        "mini rückmeldung: meh details an kritische stelle wär helpful",
        "solid arbeit, aber e chli trocken i de lesbarkeit",
        "würd eventuell visuelle element ergänze für d übersicht",
        "verständlich isch s, aber recht lang i de form",
        "würd d abschnitt e chli klarer voneinander trenne",
        "es het struktur, aber vereinzlet wirkt s chli chaotisch",
        "grundsätzlich guet, aber chli z technisch formuliert",
        "en leichtes redesign chönnt s klarer mache",
        "het substanz, aber paar stelle sind repetitiv",
        "mini empfehlig: meh praxisbezug würd helfen",
        "würd s grundsätzlich eso loh, aber verbesserbar isch s immer",
        "logisch umegsetzt, aber e chli wenig intuitiv",
        "chli kürze würd de text sicher lese-freundlicher mache",
        "ha s nüchtern analysiert – stabil, aber no luftig",
        "es isch guet, aber nöd grad herausragend",
    ],


    ("neutral", "hobby_talk"): [
        "bin grad wieder vil am gamä in de freizit",
        "go gern jogge, wenn ich dä chopf lüfte möcht",
        "ha wieder aagfange meh z läse, tut guet",
        "bin momentan voll im kochfilm und probier rezepte us",
        "lueg grad vil series zum abschalte",
        "musik lose beruhigt mi mega im alltag",
        "übe grad wieder chli uf mim instrument",
        "ha neu mit fotografie aagfange, macht spass",
        "bi gerne drausse unterwägs, wenn s wetter mitmacht",
        "versueche grad meh kreativität i mis läbe z bringe",
        "mach imfall gern chli bastel- oder diy-projekt",
        "ha wieder meh lust uf games aktuell",
        "probier grad en neue workout-routine us",
        "experimentier i de chuchi mit neue rezept",
        "bin gerne am stricke oder häkle, mega entspannend",
        "les grad en recht spannende romanreihe",
        "zeichne viel i de letzte zit, tut mer guet",
        "ha wieder zum skateboard gfunde",
        "verbringe gern zit i de natur zum abschalte",
        "cha stundenlang podcasts lose, voll mini welt",
        "lern grad öppis neus im bereich design",
        "probier grad en neue sportart us, mal luege wies wird",
        "gärtnere imfall mega gern – isch richtig chillig",
        "interessier mi grad chli für astronomie",
        "sammle seit churzem vinylplatte, mega spannend",
        "musigiere gern und improvisiere eifach druuflos",
        "entdeck grad paar alte hobbys wieder neu",
        "experimentiere imfall mit digital art und tablets",
        "puzzle gern, finds mega beruhigend",
        "ha allgemein gern alles, wo kreativität bruucht",
    ],



    ("neutral", "news_teilen"): [
        "hesch gseh, was grad i de nöi passiert isch?",
        "ich ha en spannende artikel über das thema glese",
        "es het wieder es update zu dere sach geh",
        "imfall, d app het jetzt es neus feature übercho",
        "ich ha ghört, dass sich öppis a de regel gänderet het",
        "grad im bereich git s en interessante entwicklung",
        "bin über en beitrag gstolpert, wo perfekt zum thema passt",
        "i de nöi git s grad vil diskussion drum",
        "es git en neus projekt, wo chönt spannend wärde",
        "ha gseh, dass d termi chli gschobe worde sind",
        "ich ha öppis spannends uf social media ufglese",
        "es het en bericht geh, wo ich recht interessant gfunde ha",
        "gibts es update zur situation, falls di das interessiert",
        "ha grad es kurzes video gseh, wo guet erklärt, was lauft",
        "es het en neue entwurf für die regelige geh",
        "ha grad en newsletter zue dem thema übercho",
        "im technologiebereich het s öppis neus geh",
        "bin über en artikel gstolpert, wo relevant chönnt sii",
        "ha gseh, dass en event fasch hätt müesse verschobe werde",
        "es het neue info, aber no nöd ganz bestätigt",
        "gseh, dass s projekt jetzt offiziell startet",
        "bin über en trend gstolpert, wo grad am ufchöme isch",
        "ha neus glese über öppis, wo mer im chopf blibe isch",
        "es het en update zu dere debatte geh",
        "ha gseh, dass öppis neu i de statistik ufgfüehrt worde isch",
        "git aktuell neue erkenntnisse zu dem thema",
        "ha en bericht über es zukunftsprojekt gseh",
        "ha gseh, dass öppis Grössers in planig isch",
        "es het en neue entwiklig, aber no ganz am afang",
        "ha s gfühl, da chönnt bald öppis grosses us dem usewachse",
    ],



    ("neutral", "technik_support"): [
        "mini app spinnt scho wieder und ich ha kei ahnig warum",
        "chum grad überhaupt nöd druus mit dem update",
        "mis wlan macht imfall mega komische säch grad",
        "weiss nöd genau, wie ich das programm installiere söll",
        "dä login funktioniert eifach nöd, kei plan warum",
        "ha s gfühl, mis handy isch mega langsam worde",
        "bechum ständig fehlermeldig und versteh nöd, was s bedeutet",
        "weiss nöd, welches kabel wohi ghört, komplett lost",
        "ha öppis verstellt i de einstellige und find s nüm",
        "brucht chli hilf, zum das endlich wieder zum laufe bringe",
        "mis gerät reagiert mega verzögert, voll mühsam",
        "weiss nöd, wie ich d einstellige zruggsetze cha",
        "mis programm stürzt die ganz zit ab, kei chance",
        "dä button macht nüt, egal wie oft ich druf drucke",
        "bin unsicher, öb ich s falsch installiert ha",
        "find die funktion nüm im menü, isch plötzlich verschwunde",
        "mis wlan bricht ständig ab, mega instabil",
        "weiss nöd, wie ich das update rückgängig machä cha",
        "mis gerät zeigt mega komische icons a, keine ahnig warum",
        "chan nüm uf d sache zugreife, irgendöppis blockiert",
        "weiss nöd, wie ich das dokument exportiere söll",
        "d app synchronisiert nöd, obwohl s müesst",
        "chan d datei nüm ufmache, gibt immer fehler",
        "ha systemfehler, aber kei schimmer, woher das chunnt",
        "weis nöd, wie mer das backup richtig macht",
        "mini lautsprecher funktioniere plötzlich gar nüm",
        "chan s passwort nöd zruggsetze, irgendöppis hakt",
        "weiss nöd, wie ich d verbindig wiederherstelle",
        "ha abstürz ohni ersichtlige grund, mega seltsam",
        "brucht grad ernsthaft chli technischa support, ehrlich",
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

