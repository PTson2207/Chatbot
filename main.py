from collections import Counter
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import string 
from underthesea import word_tokenize
from preprocessing import preprocess_sentence, tokenize_and_filter
MAX_LENGTH = 80
# For Transformer
NUM_LAYERS = 2
D_MODEL = 512
NUM_HEADS = 8
UNITS = 256
DROPOUT = 0.1
EPOCHS = 50

# Load data
df = pd.read_csv('data/qa_data.csv')
questions = df['questions'].dropna().to_list()
answers = df['answers'].dropna().to_list()

# Preprocess data
questions = [preprocess_sentence(question) for question in questions]
answers = [preprocess_sentence(answer) for answer in answers]

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size = 1000
)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

questions, answers = tokenize_and_filter(questions, answers)
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, : -1]
    },
    {
        'outputs': answers[:, 1: ]
    },
))
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 2000
BUFFER_SIZE = 1000
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(AUTO)

# LoadModel
