# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from prediction.make_prediction import make_prediction
from source.data_filtering import *
from spellchecker import SpellChecker

import wordninja

spell = SpellChecker()
# -------------------------------------------------------------------------
#                           Functions
# -------------------------------------------------------------------------


def infer_spaces(text):
    return " ".join(wordninja.split(text))


def filter_text(text):
    phrase = " ".join(text.split())
    phrase = convert_to_lower_case(phrase)
    phrase = contraction_mapping(phrase)
    phrase = remove_emojis(phrase)

    orig_words = phrase.split()
    return " ".join(spell.correction(word) for word in orig_words)


def test_phrase():
    phrase = input("Enter test phrase here: ").lstrip()

    phrase = filter_text(phrase)
    phrase = infer_spaces(phrase)
    print(phrase)

    result = make_prediction(phrase)
    print(result)
    print('foo')
    test_phrase()

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    test_phrase()

