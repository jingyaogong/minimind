def rep_penalty(text, tokenizer, n=3, cap=0.5):
    r"""Return a capped repetition penalty over model-token n-grams.

    Token IDs keep the calculation consistent across languages. In particular,
    contiguous CJK text must not be treated as one word, which is what happens
    with a ``\w+``-based regular expression.
    """
    if n <= 0:
        raise ValueError("n must be greater than zero")
    if cap < 0:
        raise ValueError("cap must be non-negative")

    token_ids = tokenizer.encode(text.lower(), add_special_tokens=False)
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return 0.0

    repeated = len(grams) - len(set(grams))
    return min(cap, repeated * cap * 2 / len(grams))
