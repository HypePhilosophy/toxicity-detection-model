# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
EMBEDDING_DIMENSION = 100
EMBEDDING_FILE_LOC = '../model/glove/glove.6B.' + str(EMBEDDING_DIMENSION) + 'd.txt'
TRAINING_DATA_LOC = '../data/train.csv'
TEST_DATA_LABEL = '../data/test_labels.csv'
TEST_DATA_COMMENTS = '../data/test.csv'
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 30
VALIDATION_SPLIT = 0.2
DETECTION_CLASSES = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
    'neutral']
MODEL_LOC = '../model/comments_toxicity.h5'
TOKENIZER_LOC = '../model/tokenizer.pickle'
CLEANED_TRAIN_DATA_LOC = '../exported_data/train_cleaned.csv'
EXPORTED_VECTORS_LOC = '../exported_data/exported_vectors.csv'
EXPORTED_PAD_SEQUENCES_LOC = '../exported_data/exported_pad_sequences.csv'
EXPORTED_WEIGHTS_LOC = '../exported_data/exported_weights.csv'