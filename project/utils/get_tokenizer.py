from project.utils.utils_tokenizer import select_word_based_tokenizer, CharBasedTokenizer, SplitTokenizer


def get_custom_tokenizer(lang, mode="w", prepro=True):
    """
    This function returns the tokenizer based on the configurations. The function is used either during the first preprocessing phase and during training time
    :param lang: the tokenizer language (relevant for spacy)
    :param mode: Char-based ("c") or Word-based ("w")
    :param prepro: True during the preprocessing step, False during training preprocessing
    :return: tokenizer
    """
    assert mode.lower() in ["c", "w"], "Please provide 'c' or 'w' as mode (char-level, word-level)."
    if prepro:
        mode = "w"
    if mode == "w" and prepro:
        return select_word_based_tokenizer(lang)
    elif not prepro:
        if mode == "c":
            return CharBasedTokenizer(lang)
        else:
            return SplitTokenizer(lang)