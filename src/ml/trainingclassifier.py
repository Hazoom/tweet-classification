# pylint: disable=too-many-locals
import os
import argparse

import argcomplete
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from ml import modelsaving, features
from preprocessing import cleantext

TWEET_COLUMN = 'tweet'
LABEL_COLUMN = 'label'

RANDOM_STATE = 42


def _get_model_by_name(model_type: str):
    if model_type == "LogisticRegression":
        model = LogisticRegression(random_state=RANDOM_STATE)
    elif model_type == "SVC":
        model = SVC(random_state=RANDOM_STATE)
    elif model_type == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_type == "GradientBoostingClassifier":
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif model_type == "ExtraTreesClassifier":
        model = ExtraTreesClassifier(random_state=RANDOM_STATE)
    elif model_type == 'MultinomialNB':
        model = MultinomialNB()
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

    x_train = train_df[TWEET_COLUMN].apply(cleantext.clean_tweet).to_list()

    # get model by its name
    single_model = _get_model_by_name(model_type)

    print("Vectorizing training data")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train_tfidf = vectorizer.fit_transform(x_train)

    # hyper parameters for each model
    parameters = {
        'LogisticRegression': {'penalty': ['l2'],
                               'solver': ['liblinear', 'lbfgs'],
                               'C': [1.0, 10]},
        'ExtraTreesClassifier': {'n_estimators': [16, 32]},
        'RandomForestClassifier': {'n_estimators': [16, 32]},
        'GradientBoostingClassifier': {'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0]},
        'SVC': [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
            {'kernel': ['linear'], 'C': [1, 10, 100]}
        ]
    }

    # grid search cross-validation
    clf = GridSearchCV(single_model,
                       parameters[model_type],
                       cv=5,
                       verbose=3,
                       n_jobs=-1,
                       scoring='accuracy',
                       refit=True)

    print("Training")
    clf.fit(x_train_tfidf, vec_y_cat_train)

    print("Best parameters on the validation test:")
    print(clf.best_params_)

    print("Grid scores on validation set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Saving model")
    model_dir = os.path.join(output_dir, model_type)
    modelsaving.save_model(clf.best_estimator_, model_dir)
    modelsaving.save_vectorizer(vectorizer, model_dir)
    modelsaving.save_label_encoder(label_encoder, model_dir)

    print("Predicting training set")
    predicted = clf.predict(x_train_tfidf)
    accuracy = np.mean(predicted == vec_y_cat_train)
    print("Accuracy on train set: {}".format(accuracy))

    print("Vectorizing test data")
    x_test = test_df[TWEET_COLUMN].apply(cleantext.clean_tweet).to_list()
    x_test_tfidf = vectorizer.transform(x_test)

    print("Predicting test set")
    predicted = clf.predict(x_test_tfidf)
    accuracy = np.mean(predicted == vec_y_cat_test)

    print("Accuracy on test set: {}".format(accuracy))
    target_names = [str(class_name) for class_name in label_encoder.classes_]
    print(classification_report(vec_y_cat_test,
                                predicted,
                                target_names=target_names))

    print("Plotting top K features for each class")
    features.plot_top_k_features(vectorizer, clf.best_estimator_, model_dir, k_related_terms)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-train", type=str,
                                 help='Input CSV file containing training tweets', required=True)
    argument_parser.add_argument("--input-test", type=str,
                                 help='Input CSV file containing test tweets', required=True)
    argument_parser.add_argument("--model", type=str,
                                 help='Model type to train', default='SVM', required=False)
    argument_parser.add_argument("--output-dir", type=str, help='Directory for output', required=True)
    argument_parser.add_argument("--k-related-terms", type=int,
                                 help='Number of related terms to output per company. Default: 10', required=False,
                                 default=10)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train(args.input_train, args.input_test, args.model, args.output_dir, args.k_related_terms)


if __name__ == '__main__':
    main()
