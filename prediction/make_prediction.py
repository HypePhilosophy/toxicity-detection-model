# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from source.configuration import *
from source.data_filtering import clean_text

# -------------------------------------------------------------------------
#                     Load Existing Model and Tokenizer
# -------------------------------------------------------------------------

# load the trained model
rnn_model = load_model(MODEL_LOC)

# load the tokenizer
with open(TOKENIZER_LOC, 'rb') as handle:
    tokenizer = pickle.load(handle)


# -------------------------------------------------------------------------
#                           Main Application
# -------------------------------------------------------------------------

def make_prediction(input_comment):
    """
    Predicts the toxicity of the specified comment
    :param input_comment: the comment to be verified
    """
    input_comment = clean_text(input_comment)
    input_comment = input_comment.split(" ")

    sequences = tokenizer.texts_to_sequences(input_comment)
    sequences = [[item for sublist in sequences for item in sublist]]

    padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = rnn_model.predict(padded_data, len(padded_data), verbose=1)

    return \
        {
            "Toxic": str(result[0][0]),
            "Very Toxic": str(result[0][1]),
            "Obscene": str(result[0][2]),
            "Threat": str(result[0][3]),
            "Insult": str(result[0][4]),
            "Hate": str(result[0][5]),
            "Neutral": str(result[0][6])
        }
