# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import wordninja
import nltk
import re
import spacy
import contractions
from nltk.corpus import stopwords
from textblob import TextBlob

text_count = 0
row_count = 0

# -------------------------------------------------------------------------
#                        Instance Creation
# -------------------------------------------------------------------------
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nltk.download('stopwords')
stop_words = stopwords.words('english')

# -------------------------------------------------------------------------
#                           Data Cleaning
# -------------------------------------------------------------------------


def convert_to_lower_case(text):
    return " ".join(text.lower() for text in text.split())


def contraction_mapping(text):
    special_characters = ["’", "‘", "´", "`"]
    for s in special_characters:
        text = text.replace(s, "'")
    text = ' '.join(contractions.fix(text) for text in text.split(" "));
    return text;

def fix_misspelled_words(text):
    """
    Fixes the misspelled words on the specified text (uses TextBlob model)
    :param text: The text to be fixed
    :return: the fixed text
    """
    b = TextBlob(text)
    return str(b.correct())

def fix_misspelled_words2(text):
    """
    Fixes the misspelled words on the specified text (uses predefined misspelled dictionary)
    :param text: The text to be fixed
    :return: the fixed text
    """
    mispelled_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                      'counselling': 'counseling',
                      'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization',
                      'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora',
                      'sallary': 'salary',
                      'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are',
                      'howcan': 'how can',
                      'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                      'theBest': 'the best',
                      'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate',
                      "mastrubating": 'masturbating',
                      'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data',
                      '2k17': '2017', '2k18': '2018',
                      'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',
                      'watsapp': 'whatsapp',
                      'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                      'demonetisation': 'demonetization', ' ur ': 'your', ' u r ': 'you are'}
    for word in mispelled_dict.keys():
        text = text.replace(word, mispelled_dict[word])
    return text

def remove_punctuations(text):
    """
    Removes all punctuations from the specified text
    :param text: the text whose punctuations to be removed
    :return: the text after removing the punctuations
    """
    return text.replace(r'[^\w\s]', '')


def remove_emojis(text):
    """
    Removes emojis from the specified text
    :param text: the text whose emojis need to be removed
    :return: the text after removing the emojis
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_stopwords(text):
    """
    Removes all stop words from the specified text
    :param text: the text whose stop words need to be removed
    :return: the text after removing the stop words
    """
    return " ".join(x for x in text.split() if x not in stop_words)


def lemmatise(text):
    """
    Lemmatises the specified text
    :param text: the text which needs to be lemmatised
    :return: the lemmatised text
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def infer_spaces(text):
    return " ".join(wordninja.split(text))


def clean_text_column(text_column):
    """
    Cleans the text specified in the text column
    The cleaning procedure is as follows:
    1. Convert the context to lower case
    2. Apply contraction mapping in which we fix shorter usages of english sentences
    3. Fix misspelled words
    4. Remove all punctuations
    5. Remove all emojis
    6. Remove all stop words
    7. Apply Lemmatisation
    :param text_column: the text column which needs to be cleaned
    :return: the text column with the cleaned data
    """

    global row_count
    row_count = len(text_column.index)

    return text_column.apply(lambda x: clean_text(x))


def clean_text(text):
    global text_count

    """
    Cleans the specified text
    The cleaning procedure is as follows:
    1. Convert the context to lower case
    2. Apply contraction mapping in which we fix shorter usages of english sentences
    3. Fix misspelled words
    4. Remove all punctuations
    5. Remove all emojis
    6. Remove all stop words
    7. Apply Lemmatisation
    :param text: the text which needs to be cleaned
    :return: the cleaned text
    """
    text = convert_to_lower_case(text)
    text = contraction_mapping(text)
    text = fix_misspelled_words2(text)
    text = remove_punctuations(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    text = lemmatise(text)

    text_count += 1
    print(f'Cleaned Word Count: {text_count}/{row_count}', end='\r')

    return text

