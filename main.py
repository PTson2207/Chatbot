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
from model import transformer
from evaluate import loss_function, accuracy, CustomSchedule, evaluate, predict


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
tf.keras.backend.clear_session()

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

with strategy.scope():
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

#model.fit(dataset, epochs=500)
model.load_weights('transformer_chatbot.h5')


while (1):
    question = input("Question: ")
    if question == 'quit':
        break
    sentence = predict(question)