import argparse
import pandas as pd
import argcomplete

from ml.classifier import Classifier


def predict_tweets(tweets_file: str,
                   output_file: str,
                   classifier_dir: str) -> None:
    print('Reading file...')
    tweets_df = pd.read_csv(open(tweets_file, 'rU'), encoding='utf-8', engine='c', delimiter='\t',
                            header=None, names=['id', 'tweet'])
    print(f'No. of tweets: {len(tweets_df)}')

    classifier = Classifier(classifier_dir)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--tweets-file", type=str, help='Tweets TXT file', required=True)
    argument_parser.add_argument("--output-file", type=str, help='Output titles txt', required=True)
    argument_parser.add_argument("--classifier-dir", type=str, help='Classifier directory', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    predict_tweets(args.tweets_file, args.output_file, args.classifier_dir)


if __name__ == '__main__':
    main()
