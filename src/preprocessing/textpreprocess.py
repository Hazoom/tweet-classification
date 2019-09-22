# pylint: disable=too-many-instance-attributes
import argparse
import os
from collections import OrderedDict
from typing import List, Tuple

import argcomplete
import dill
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from preprocessing import cleantext


class TextPreprocessor:
    def __init__(self,
                 n_vocab: int = 10000,
                 max_length: int = 300,
                 truncating: str = 'post',
                 padding: str = 'pre'):
        self.padding = padding
        self.truncating = truncating
        self.max_length = max_length
        self.n_vocab = n_vocab
        self.tokenizer = text_to_word_sequence
        self.cleaner = cleantext.clean_tweet
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.token_to_id = None
        self.id_to_token = None
        self.n_tokens = None
        self.padding_value = 0
        self.indexer = None

    def process_texts(self,
                      texts: List[str]):
        return [self.tokenizer(self.cleaner(doc)) for doc in texts]

    def transform(self, texts: List[str]):
        if self.token_to_id is None:
            raise Exception('Model is not fitted yet!')

        tokenized_data = self.process_texts(texts)

        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        padded_sequences = self._pad_sequences(indexed_data)

        return padded_sequences

    def _transform_tokenized_texts(self, tokenized_data):
        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        padded_sequences = self._pad_sequences(indexed_data)

        return padded_sequences

    def fit_transform(self, texts: List[str]):
        tokenized_data = self.fit(texts)
        return self._transform_tokenized_texts(tokenized_data)

    def fit(self, texts: List[str]):
        print('Tokenize...')
        tokenized_texts = self.process_texts(texts)
        assert len(tokenized_texts) == len(texts)
        print('Done tokenize')

        self.indexer = CustomIndexer(num_words=self.n_vocab)

        print('Fitting...')
        self.indexer.fit_on_tokenized_texts(tokenized_texts)
        print('Done fitting')

        # Build Dictionary accounting For 0 padding, and reserve 1 for unknown and rare words
        self.token_to_id = self.indexer.word_index
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.n_tokens = max(self.indexer.word_index.values())

        return tokenized_texts

    def _pad_sequences(self, sequences):
        return pad_sequences(sequences,
                             maxlen=self.max_length,
                             padding=self.padding,
                             truncating=self.truncating,
                             value=self.padding_value)


class CustomIndexer(Tokenizer):
    """
    Text vectorization utility class.
    This class inherits keras.preprocess.text.Tokenizer but adds methods
    to fit and transform on already tokenized text.
    Parameters
    ----------
    num_words : int
        the maximum number of words to keep, based
        on word frequency. Only the most common `num_words` words will
        be kept.
    """

    def __init__(self, num_words, **kwargs):
        # super().__init__(num_words, **kwargs)
        self.num_words = num_words
        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.document_count = 0

    def fit_on_tokenized_texts(self, tokenized_texts):
        self.document_count = 0
        for seq in tokenized_texts:
            self.document_count += 1
            for word in seq:
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
            for word in set(seq):
                if word in self.word_docs:
                    self.word_docs[word] += 1
                else:
                    self.word_docs[word] = 1

        word_counts = list(self.word_counts.items())
        word_counts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in word_counts][:self.num_words]
        # note that index 0 and 1 are reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(2, len(sorted_voc) + 2)))))

    def tokenized_texts_to_sequences(self, tok_texts):
        """Transforms tokenized text to a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            tokenized texts:  List[List[str]]
        # Returns
            A list of integers.
        """
        res = []
        for vector in self.tokenized_texts_to_sequences_generator(tok_texts):
            res.append(vector)
        return res

    def tokenized_texts_to_sequences_generator(self, tok_texts):
        """Transforms tokenized text to a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            tokenized texts:  List[List[str]]
        # Yields
            Yields individual sequences.
        """
        for seq in tok_texts:
            vector = []
            for word in seq:
                # if the word is missing you get oov_index
                index = self.word_index.get(word, 1)
                vector.append(index)
            yield vector


def load_text_preprocessor(file_path: str) -> Tuple[TextPreprocessor, int]:
    """
    Load TextPreprocessor dpickle file from disk.
    :param file_path: str
        File path on disk
    :return:
    text_pre_processor: TextPreprocessor
    n_tokens: int
    """
    with open(file_path, 'rb') as in_fp:
        text_pre_processor = dill.load(in_fp)
    n_tokens = text_pre_processor.n_tokens + 1  # + 1 because of padding token
    print(f'Loaded model: {file_path}. Number of tokens: {n_tokens}')
    return text_pre_processor, n_tokens


def _save_pre_processor(pre_processor: TextPreprocessor, output_dir: str, file_name: str):
    with open(os.path.join(output_dir, file_name), 'wb+') as out_fp:
        dill.dump(pre_processor, out_fp)


def _save_vectors(vectors, output_dir: str, file_name: str):
    np.save(os.path.join(output_dir, file_name), vectors)


def parse_data(train_input_file: str,
               test_input_file: str,
               output_dir: str):
    train_df = pd.read_csv(train_input_file)
    tweet_pre_processor = TextPreprocessor(n_vocab=10000, max_length=300, truncating='post', padding='post')

    print('Fitting pre-processor on tweets...')
    train_tweet_vectors = tweet_pre_processor.fit_transform(train_df['tweet'].tolist())
    print('Finished fitting pre-processor on tweets')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Saving training pre processors and vectors...')

    # save the pre processors for later use
    _save_pre_processor(tweet_pre_processor, output_dir, 'tweet_pre_processor.dpkl')

    # save the training vectors
    _save_vectors(train_tweet_vectors, output_dir, 'train_tweets_vectors.npy')

    test_df = pd.read_csv(test_input_file)

    print('Transforming test tweets...)')
    test_tweet_vectors = tweet_pre_processor.transform(test_df['tweet'].tolist())
    print('Finished transforming test tweets')

    # save the test vectors of tweets
    _save_vectors(test_tweet_vectors, output_dir, 'test_tweets_vectors.npy')

    print('Done.')


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train-input-file", type=str, help='Input training CSV file',
                                 required=True)
    argument_parser.add_argument("--test-input-file", type=str, help='Input test CSV file',
                                 required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    parse_data(args.train_input_file, args.test_input_file, args.output_dir)


if __name__ == '__main__':
    main()
