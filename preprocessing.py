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

df = pd.read_csv('data/qa_data.csv')
#print(df.shape)
questions = df['questions'].dropna().to_list()
answers = df['answers'].dropna().to_list()

def preprocess_sentence(text):
    text = ''.join([i for i in text if i not in string.punctuation])
    return word_tokenize(text.strip(), format='text').lower()

questions = [preprocess_sentence(question) for question in questions]
answers = [preprocess_sentence(answer) for answer in answers]

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size = 1000
)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

MAX_LENGTH = 80
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs = []
    tokenized_outputs = []
    for (input_sent, output_sent) in zip(inputs, outputs):
        # mã hóa câu
        input_sent = START_TOKEN + tokenizer.encode(input_sent) + END_TOKEN
        output_sent = START_TOKEN + tokenizer.encode(output_sent) + END_TOKEN
        # kiểm tra câu mã hóa với chiều dài qui định
        if len(input_sent) <= MAX_LENGTH and len(output_sent) <= MAX_LENGTH:
            tokenized_inputs.append(input_sent)
            tokenized_outputs.append(output_sent)

    # pad sentence
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post'
    )
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post'
    )
    return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)

print('Vocab size: {}'.format(VOCAB_SIZE))
print('Number of Sample questions:{}'.format(len(questions)))
print('Number of Sample answers: {}'.format(len(answers)))