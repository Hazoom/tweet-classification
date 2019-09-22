import argparse
import os

import argcomplete
import numpy as np
from sklearn.metrics import classification_report

from dl.attention_bi_lstm import build_model, load_labels, load_vectors
from preprocessing import textpreprocess

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


def evaluate(test_file: str,
             test_vectors_file: str,
             tweet_preprocessor_file: str,
             model_dir: str,
             learning_rate: float = 0.0005,
             word_embedding_dim: int = 300,
             hidden_state_dim: int = 128):
    x_test = load_vectors(test_vectors_file)

    tweet_pre_processor, vocab_size = textpreprocess.load_text_preprocessor(tweet_preprocessor_file)

    y_test = load_labels(test_file)

    assert x_test.shape[0] == len(y_test)

    model = build_model(x_test.shape[1],
                        word_embedding_dim,
                        vocab_size,
                        hidden_state_dim,
                        learning_rate=learning_rate)

    model.load_weights(os.path.join(model_dir, 'attention_bi_lstm_model.h5'))

    print('Evaluating on test set...')
    y_pred = model.predict(x_test)
    y_pred = (y_pred >= 0.5).astype(np.int)
    report = classification_report(y_test, y_pred, target_names=['Non Marketing', 'Marketing'])
    print('Classification Report:\n', report, '\n')


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--test-file", type=str, help='Test tweets file path', required=True)
    argument_parser.add_argument("--test-vectors-file", type=str, help='Test vectors file path', required=True)
    argument_parser.add_argument("--tweet-preprocessor-file", type=str,
                                 help='Train pre processor file path', required=True)
    argument_parser.add_argument("--model-dir", type=str, help='Directory for model', required=True)
    argument_parser.add_argument("--learning-rate", type=float, help='Learning rate. Default: 0.005',
                                 required=False, default=0.005)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    evaluate(args.test_file,
             args.test_vectors_file,
             args.tweet_preprocessor_file,
             args.model_dir,
             args.learning_rate)


if __name__ == '__main__':
    main()
