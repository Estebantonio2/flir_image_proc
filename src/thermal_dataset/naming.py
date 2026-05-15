SPANISH_TO_ENGLISH = {
    "vidrio": "glass",
    "madera": "wood",
    "mano": "hand",
    "derecha": "right",
    "izquierda": "left",
    "superficie": "surface",
    "pre": "pre",
    "test": "test",
}


def translate_token(value: str) -> str:
    normalized = value.strip().lower()
    return SPANISH_TO_ENGLISH.get(normalized, normalized)


def translate_compound_name(value: str) -> str:
    separators = ["_", "-"]
    translated = value

    for separator in separators:
        if separator in translated:
            return separator.join(translate_compound_name(part) for part in translated.split(separator))

    return translate_token(translated)
