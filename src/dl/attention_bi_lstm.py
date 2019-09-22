import argparse
import os

import argcomplete
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Concatenate

from dl.attention import Attention
from preprocessing import textpreprocess

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


def load_vectors(file_path: str):
    code_vectors = np.load(file_path)
    print(f'Shape of vectors: {code_vectors.shape}')
    return code_vectors


def load_labels(csv_file: str):
    tweets = pd.read_csv(csv_file)

    # convert labels to their matching index
    return tweets['label'].apply(lambda label: 0 if label == 'Marketing' else 1).tolist()


def train(train_file: str, test_file: str,
          train_vectors_file: str, test_vectors_file: str,
          tweet_preprocessor_file: str,
          output_dir: str,
          epochs: int = 16,
          batch_size: int = 32,
          validation_split: float = 0.15,
          learning_rate: float = 0.0005,
          word_embedding_dim: int = 300,
          hidden_state_dim: int = 128):
    x_train = load_vectors(train_vectors_file)
    x_test = load_vectors(test_vectors_file)

    tweet_pre_processor, vocab_size = textpreprocess.load_text_preprocessor(tweet_preprocessor_file)

    y_train = load_labels(train_file)
    y_test = load_labels(test_file)

    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)

    model = build_model(x_train.shape[1],
                        word_embedding_dim,
                        vocab_size,
                        hidden_state_dim,
                        learning_rate=learning_rate)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            min_delta=0,
                                                            patience=1,
                                                            verbose=0, mode='auto')

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(output_dir, 'model_weights_best.hdf5'),
        save_best_only=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              callbacks=[early_stopping_callback, model_checkpoint])

    model.save(os.path.join(output_dir, 'attention_bi_lstm_model.h5'))


def build_model(seq_len: int,
                word_embedding_dim: int,
                vocab_size: int,
                hidden_state_dim: int,
                learning_rate: float):
    sequence_input = Input(shape=(seq_len,), dtype='int32')
    embedded_sequences = keras.layers.Embedding(vocab_size, word_embedding_dim, input_length=seq_len)(sequence_input)

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                         (hidden_state_dim,
                                          dropout=.3,
                                          recurrent_dropout=.4,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_activation='relu',
                                          recurrent_initializer='glorot_uniform'),
                                         name="bi_lstm_0")(embedded_sequences)

    lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_state_dim,
                             dropout=0.2,
                             recurrent_dropout=.4,
                             return_sequences=True,
                             return_state=True,
                             recurrent_activation='relu',
                             recurrent_initializer='glorot_uniform'),
        name='bi_lstm_1')(lstm)

    state_h = Concatenate()([forward_h, backward_h])

    attention = Attention(hidden_state_dim)

    context_vector, attention_weights = attention(lstm, state_h)

    output = keras.layers.Dense(1, activation='sigmoid')(context_vector)

    model = keras.Model(inputs=sequence_input, outputs=output, name="TweetsModel")

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Nadam(lr=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train-file", type=str, help='Train tweets file path', required=True)
    argument_parser.add_argument("--test-file", type=str, help='Test tweets file path', required=True)
    argument_parser.add_argument("--train-vectors-file", type=str, help='Train vectors file path', required=True)
    argument_parser.add_argument("--test-vectors-file", type=str, help='Test vectors file path', required=True)
    argument_parser.add_argument("--tweet-preprocessor-file", type=str,
                                 help='Train pre processor file path', required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory for model', required=True)
    argument_parser.add_argument("--epochs", type=int, help='Number of epochs. Default: 16', required=False,
                                 default=16)
    argument_parser.add_argument("--batch-size", type=int, help='Batch size. Default: 32', required=False,
                                 default=32)
    argument_parser.add_argument("--validation-split", type=float, help='Validation size. Default: 0.15',
                                 required=False, default=0.15)
    argument_parser.add_argument("--learning-rate", type=float, help='Learning rate. Default: 0.005',
                                 required=False, default=0.005)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train(args.train_file, args.test_file,
          args.train_vectors_file, args.test_vectors_file,
          args.tweet_preprocessor_file,
          args.output_dir,
          args.epochs, args.batch_size, args.validation_split, args.learning_rate)


if __name__ == '__main__':
    main()
