# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from flask import Flask, request
from flask_restful import Resource, Api
from prediction.make_prediction import make_prediction
from source.data_filtering import *
from spellchecker import SpellChecker
import wordninja

# -------------------------------------------------------------------------
#                           Functions
# -------------------------------------------------------------------------
# Initialize App
app = Flask(__name__)
api = Api(app)
spell = SpellChecker()


def infer_spaces(text):
    return " ".join(wordninja.split(text))


def filter_text(text):
    phrase = " ".join(text.split())
    phrase = convert_to_lower_case(phrase)
    phrase = contraction_mapping(phrase)
    phrase = remove_emojis(phrase)

    orig_words = phrase.split()
    return " ".join(spell.correction(word) for word in orig_words)


class Predictions(Resource):

    def post(self):
        # Get the data from the request
        text = request.form['text']
        phrase = text.lstrip()

        phrase = filter_text(phrase)
        phrase = infer_spaces(phrase)
        print(phrase)

        result = make_prediction(phrase)
        print(result)
        return {'data': result}, 200

    pass


api.add_resource(Predictions, '/predictions')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
