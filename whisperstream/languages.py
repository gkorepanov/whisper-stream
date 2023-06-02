from iso639 import Lang

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
    return _NAME_TO_LANG[name.capitalize()]
