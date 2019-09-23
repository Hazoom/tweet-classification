import argparse
import pandas as pd
import argcomplete
from pathos.multiprocessing import cpu_count
from pypeln import thread as th

from ml.classifier import Classifier


def _get_prediction(classifier, row, index, predicted, total_length_str):
    prediction = classifier.predict(str(row['tweet']))

    predicted.append((index, prediction))

    if index % 1000 == 0 and index > 0:
        print(f'Finished {str(index)} out of {total_length_str}')


def predict_tweets(tweets_file: str,
                   output_file: str,
                   classifier_dir: str,
                   limit: int) -> None:
    print('Reading file...')
    tweets_df = pd.read_csv(open(tweets_file, 'rU'), encoding='utf-8', engine='c', delimiter='\t',
                            header=None, names=['id', 'tweet'])
    tweets_df = tweets_df.head(limit)
    print(f'No. of tweets: {len(tweets_df)}')

    classifier = Classifier(classifier_dir)

    # predict tweets with multiple threads
    cpu_cores = cpu_count()
    predictions = []
    total_length_str = str(len(tweets_df))
    (tweets_df.iterrows()
     | th.each(lambda x: _get_prediction(classifier, x[1], x[0], predictions, total_length_str),
               workers=cpu_cores, maxsize=0)
     | list)

    print(f'Finished {total_length_str} out of {total_length_str}')

    # sort results by index
    print("Sorting results...")
    predictions = sorted(predictions, key=lambda key: key[0])

    # take only the prediction
    predictions = [prediction[1] for prediction in predictions]

    # add the prediction column
    print("Add prediction column to dataframe...")
    tweets_df['Prediction'] = predictions

    # write predictions
    print("Write results...")
    with open(output_file, 'w+') as out_fp:
        tweets_df.to_csv(out_fp)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--tweets-file", type=str, help='Tweets TXT file', required=True)
    argument_parser.add_argument("--output-file", type=str, help='Output titles txt', required=True)
    argument_parser.add_argument("--classifier-dir", type=str, help='Classifier directory', required=True)
    argument_parser.add_argument("--limit", type=str, help='Limit for tweets', required=False,
                                 default=100000)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    predict_tweets(args.tweets_file, args.output_file, args.classifier_dir, args.limit)


if __name__ == '__main__':
    main()
