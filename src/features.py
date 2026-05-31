def extract_features(text):
    text_lower = text.lower()
    
    has_url = int("url_token" in text_lower)
    is_suspicious_url = 0
    if has_url:
        
        if len(text_lower) > 100 or text_lower.count('.') > 3:
            is_suspicious_url = 1

    return [
        has_url,
        is_suspicious_url, 
        int("urgent" in text_lower),
        int("verify" in text_lower),
        int("select" in text_lower),
        int("drop" in text_lower),
        int("sql_comment" in text_lower),
        int("=" in text_lower),
    ]