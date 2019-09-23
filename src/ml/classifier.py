import json
import os
import pickle

import joblib
from preprocessing import cleantext


class Classifier:
    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, 'model.joblib')
        print("Loading model: {}".format(model_path))
        self.model = joblib.load(model_path)
        print("Finished loading model: {}".format(model_path))
        self.decoder_dict = {}
        labels_dict = dict(json.load(open(os.path.join(model_dir, 'label_encoder.json'), 'r')))
        for name, key in labels_dict.items():
            self.decoder_dict[key] = name
        self.vectorizer = pickle.load(open(os.path.join(model_dir, 'vectorizer.pickle'), 'rb'))

    def predict(self, x_test):
        x_test = cleantext.clean_tweet(x_test)
        vectorizerd_text = self.vectorizer.transform([x_test])
        prediction = self.model.predict(vectorizerd_text)[0]
        return self.decoder_dict[prediction]
