# These codes are from the codes for
# Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference by
# Tom McCoy, Ellie Pavlick, Tal Linzen, ACL 2019

def have_lexical_overlap(premise, hypothesis, get_hans_new_features=False):
    prem_words = []
    hyp_words = []

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    all_in = True

    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break

    if get_hans_new_features:
        overlap_percent = len(list(set(hyp_words) & set(prem_words))) / len(set(hyp_words))
    else:
        overlap_percent = len(list(set(hyp_words) & set(prem_words))) / len(set(prem_words))

    return all_in, overlap_percent


def is_subsequence(premise, hypothesis):
    prem_words = []
    hyp_words = []

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())

    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)

    return hyp_filtered in prem_filtered


def parse_phrase_list(parse, phrases):
    if parse == "":
        return phrases

    phrase_list = phrases

    words = parse.split()
    this_phrase = []
    next_level_parse = []
    for index, word in enumerate(words):
        if word == "(":
            next_level_parse += this_phrase
            this_phrase = ["("]

        elif word == ")" and len(this_phrase) > 0 and this_phrase[0] == "(":
            phrase_list.append(" ".join(this_phrase[1:]))
            next_level_parse += this_phrase[1:]
            this_phrase = []
        elif word == ")":
            next_level_parse += this_phrase
            next_level_parse.append(")")
            this_phrase = []
        else:
            this_phrase.append(word)
    return parse_phrase_list(" ".join(next_level_parse), phrase_list)


def is_constituent(premise, hypothesis, parse):
    parse_new = []
    for word in parse.split():
        if word not in [".", "?", "!"]:
            parse_new.append(word.lower())

    all_phrases = parse_phrase_list(" ".join(parse_new), [])

    prem_words = []
    hyp_words = []

    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower().replace(".", "").replace("?", "").replace("!", ""))

    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower().replace(".", "").replace("?", "").replace("!", ""))

    hyp_filtered = " ".join(hyp_words)
    return hyp_filtered in all_phrases
