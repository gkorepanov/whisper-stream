from iso639 import Lang
from whisperstream.error import UnsupportedLanguageError

# langs from https://help.openai.com/en/articles/7031512-whisper-api-faq
_WHISPER_LANGUAGES = [
    ('af', 'Afrikaans'),
    ('ar', 'Arabic'),
    ('hy', 'Armenian'),
    ('az', 'Azerbaijani'),
    ('be', 'Belarusian'),
    ('bs', 'Bosnian'),
    ('bg', 'Bulgarian'),
    ('ca', 'Catalan'),
    ('zh', 'Chinese'),
    ('hr', 'Croatian'),
    ('cs', 'Czech'),
    ('da', 'Danish'),
    ('nl', 'Dutch'),
    ('en', 'English'),
    ('et', 'Estonian'),
    ('fi', 'Finnish'),
    ('fr', 'French'),
    ('gl', 'Galician'),
    ('de', 'German'),
    ('he', 'Hebrew'),
    ('hi', 'Hindi'),
    ('hu', 'Hungarian'),
    ('is', 'Icelandic'),
    ('id', 'Indonesian'),
    ('it', 'Italian'),
    ('ja', 'Japanese'),
    ('kn', 'Kannada'),
    ('kk', 'Kazakh'),
    ('ko', 'Korean'),
    ('lv', 'Latvian'),
    ('lt', 'Lithuanian'),
    ('mk', 'Macedonian'),
    ('mr', 'Marathi'),
    ('mi', 'Maori'),
    ('no', 'Norwegian'),
    ('fa', 'Persian'),
    ('pl', 'Polish'),
    ('pt', 'Portuguese'),
    ('ro', 'Romanian'),
    ('ru', 'Russian'),
    ('sr', 'Serbian'),
    ('sk', 'Slovak'),
    ('sl', 'Slovenian'),
    ('es', 'Spanish'),
    ('sv', 'Swedish'),
    ('tl', 'Tagalog'),
    ('ta', 'Tamil'),
    ('th', 'Thai'),
    ('tr', 'Turkish'),
    ('uk', 'Ukrainian'),
    ('ur', 'Urdu'),
    ('vi', 'Vietnamese'),
    ('el', 'Greek'),
    ('ms', 'Malay'),
    ('ne', 'Nepali'),
    ('sw', 'Swahili'),
    ('cy', 'Welsh')
]

SUPPORTED_LANGUAGES = [Lang(lang) for lang, _ in _WHISPER_LANGUAGES]


_LANG_TO_NAME = {Lang(lang): name for lang, name in _WHISPER_LANGUAGES}
_NAME_TO_LANG = {name: Lang(lang) for lang, name in _WHISPER_LANGUAGES}


def get_lang_name(lang: Lang) -> str:
    return _LANG_TO_NAME[lang]


def get_lang_from_name(name: str) -> Lang:
    if not name:
        raise UnsupportedLanguageError("Language name cannot be empty.")
    try:
        return _NAME_TO_LANG[name.capitalize()]
    except KeyError:
        raise UnsupportedLanguageError(f"Language {name} is not supported.")


PUNCTUATION_PROMPTS_BY_LANG = {
    'af': "Wel, wat wou ek sê? Kom ons begin.",
    'ar': "حسنًا ، ماذا أردت أن أقول؟ هيا بنا نبدأ.",
    'hy': "Լավ, ի՞նչ էի ցանկանում ասել. Եկեք սկսենք:",
    'az': "Yaxşı, nə demək istədim? Başlayaq.",
    'be': "Ну, што я хацеў сказаць? Давайце пачнем.",
    'bs': "Pa, šta sam htio reći? Hajde da počnemo.",
    'bg': "Добре, какво исках да кажа? Нека започнем.",
    'ca': "Bé, què volia dir? Comencem.",
    'zh': "好吧，我想说什么？我们开始吧。",
    'hr': "Pa, što sam htio reći? Hajde da počnemo.",
    'cs': "No, co jsem chtěl říct? Začněme.",
    'da': "Nå, hvad ville jeg sige? Lad os begynde.",
    'nl': "Nou, wat wilde ik zeggen? Laten we beginnen.",
    'en': "Well, what did I want to say? Let's start.",
    'et': "Noh, mida ma öelda tahtsin? Alustame.",
    'fi': "No, mitä halusin sanoa? Aloittakaamme.",
    'fr': "Eh bien, que voulais-je dire? Commençons.",
    'gl': "Beno, que quería dicir? Comecemos.",
    'de': "Nun, was wollte ich sagen? Lassen Sie uns anfangen.",
    'he': "ובכן, מה רציתי להגיד? בוא נתחיל.",
    'hi': "अच्छा, मैं क्या कहना चाहता था? चलो शुरू करते हैं।",
    'hu': "Nos, mit akartam mondani? Kezdjük.",
    'is': "Já, hvað vildi ég segja? Byrjumst.",
    'id': "Nah, apa yang ingin saya katakan? Mari kita mulai.",
    'it': "Bene, cosa volevo dire? Iniziamo.'",
    'ja': "まあ、何を言いたかったのか？始めましょう。",
    'kn': "ಹೌದು, ನಾನು ಏನು ಹೇಳಲು ಬಯಸಿದೆನು? ಪ್ರಾರಂಭಿಸೋಣ.",
    'kk': "Жақсы, мен не деймекші едім? Бастау керек.",
    'ko': "그래, 내가 뭐라고 하고 싶었지? 시작합시다.",
    'lv': "Nu, ko es gribēju teikt? Sāksim.",
    'lt': "Na, ką norėjau pasakyti? Pradėkime.",
    'mk': "Добро, што сакав да кажам? Ајде да започнеме.",
    'mr': "चांगले, मला काय म्हणायचे होते? चालू करूया.",
    'mi': "Na, ka aha ahau e korero ana? Ka tiimata tatou.",
    'no': "Vel, hva ville jeg si? La oss starte.",
    'fa': "خب ، چه می خواستم بگویم؟ بیایید شروع کنیم.",
    'pl': "No dobrze, co chciałem powiedzieć? Zacznijmy.",
    'pt': "Bem, o que eu queria dizer? Vamos começar.",
    'ro': "Ei bine, ce voiam sa spun? Hai sa incepem.",
    'ru': "Ну, что я хотел сказать? Давайте начнем.",
    'sr': "Па, шта сам хтео рећи? Хајде да почнемо.",
    'sk': "No, čo som chcel povedať? Poďme začať.",
    'sl': "No, kaj sem hotel reči? Začnimo.",
    'es': "Bueno, ¿qué quería decir? Empecemos.",
    'sv': "Nåväl, vad ville jag säga? Låt oss börja.",
    'tl': "Abah, ano ba ang gusto kong sabihin? Simulan na natin.",
    'ta': "நலமாக, நான் என்ன சொல்ல வேண்டியதாக இருந்தது? ஆரம்பிப்போம்.",
    'th': "เยี่ยม, ฉันต้องการจะพูดอะไร? เริ่มเถอะ",
    'tr': "Peki, ne demek istedim? Hadi başlayalım.",
    'uk': "Ну, що я хотів сказати? Давайте почнемо.",
    'ur': "اچھا ، میں کیا کہنا چاہتا تھا؟ آئیے شروع کریں۔",
    'vi': "Vậy, tôi muốn nói gì? Hãy bắt đầu.",
    'el': "Λοιπόν, τι ήθελα να πω; Ας ξεκινήσουμε.",
    'ms': "Baiklah, apa yang saya mahu katakan? Mari kita mula.",
    'ne': "हुन्न, म भन्न चाहन्थें? सुरु गरौं।",
    'sw': "Vizuri, nataka kusema nini? Hebu tuanze.",
    'cy': "Wel, beth oeddwn i am ei ddweud? Gadewch inni ddechrau."
}

def get_punctuation_prompt_for_lang(lang: Lang) -> str:
    return PUNCTUATION_PROMPTS_BY_LANG[lang.pt1]