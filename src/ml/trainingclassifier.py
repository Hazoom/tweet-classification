# pylint: disable=too-many-locals
import os
import argparse

import argcomplete
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from ml import modelsaving, features

TWEET_COLUMN = 'tweet'
LABEL_COLUMN = 'label'

RANDOM_STATE = 42


def _get_model_by_name(model_type: str):
    if model_type == "Logistic Regression":
        model = LogisticRegression(random_state=RANDOM_STATE,
                                   solver='lbfgs',
                                   multi_class='multinomial',
                                   warm_start=True)
    elif model_type == "SVM":
        model = SVC(kernel='rbf', gamma='scale')
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "SGD":
        model = SGDClassifier(loss='log', penalty='l2', random_state=RANDOM_STATE)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    elif model_type == "Extra Trees":
        model = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif model_type == 'MultinomialNB':
        model = MultinomialNB()
    elif model_type == 'ComplementNB':
        model = ComplementNB()
    else:
        raise ValueError("Unknown model type")
    print("Model type is: {}".format(model_type))
    return model


def train(input_train_csv: str,
          input_test_csv: str,
          model_type: str,
          output_dir: str,
          k_related_terms: str) -> None:
    print("Reading files")
    train_df = pd.read_csv(input_train_csv)
    test_df = pd.read_csv(input_test_csv)

    print("Encoding labels")
    y_train = train_df[LABEL_COLUMN].to_list()
    labels = list(set(y_train))
    y_test = test_df[LABEL_COLUMN].to_list()
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    vec_y_cat_train = label_encoder.transform(y_train)
    vec_y_cat_test = label_encoder.transform(y_test)

    print("Reading training files")
    x_train = train_df[TWEET_COLUMN].to_list()

    # get model by its name
    model = _get_model_by_name(model_type)

    print("Vectorizing training data")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, min_df=5)
    x_train_tfidf = vectorizer.fit_transform(x_train)

    print("Trainig")
    model.fit(x_train_tfidf, vec_y_cat_train)

    print("Saving model")
    model_dir = os.path.join(output_dir, model_type)
    modelsaving.save_model(model, model_dir)
    modelsaving.save_vectorizer(vectorizer, model_dir)
    modelsaving.save_label_encoder(label_encoder, model_dir)

    print("Predicting training set")
    predicted = model.predict(x_train_tfidf)
    accuracy = np.mean(predicted == vec_y_cat_train)
    print("Accuracy on train set: {}".format(accuracy))

    print("Vectorizing test data")
    x_test = test_df[TWEET_COLUMN].to_list()
    x_test_tfidf = vectorizer.transform(x_test)

    print("Predicting test set")
    predicted = model.predict(x_test_tfidf)
    accuracy = np.mean(predicted == vec_y_cat_test)

    print("Accuracy on test set: {}".format(accuracy))
    target_names = [str(class_name) for class_name in label_encoder.classes_]
    print(classification_report(vec_y_cat_test,
                                predicted,
                                target_names=target_names))

    print("Saving top K features for each class")
    # features.save_top_k_features(vectorizer, model, model_dir, label_encoder, k_related_terms)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-train", type=str,
                                 help='Input CSV file containing training tweets', required=True)
    argument_parser.add_argument("--input-test", type=str,
                                 help='Input CSV file containing test tweets', required=True)
    argument_parser.add_argument("--model", type=str,
                                 help='Model type to train: SVM|Logistic Regression|Linear Regression|Random '
                                      'Forest|Decision Tree',
                                 default='SVM',
                                 required=False)
    argument_parser.add_argument("--output-dir", type=str, help='Directory for output', required=True)
    argument_parser.add_argument("--k-related-terms", type=int,
                                 help='Number of related terms to output per company. Default: 10', required=False,
                                 default=10)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train(args.input_train, args.input_test, args.model, args.output_dir, args.k_related_terms)


if __name__ == '__main__':
    main()
