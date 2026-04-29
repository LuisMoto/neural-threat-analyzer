def extract_features(text):
    text_lower = text.lower()

    return [
        int("url_token" in text_lower),
        int("urgent" in text_lower),
        int("verify" in text_lower),
        int("select" in text_lower),
        int("drop" in text_lower),
        int("sql_comment" in text_lower),
        int("=" in text_lower),
    ]