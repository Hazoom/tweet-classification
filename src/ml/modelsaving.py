import json
import os
import pickle

import joblib


def maybe_create_model_dir(model_dir: str) -> None:
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)


def get_classes_dictionary(label_encoder) -> dict:
    return dict(zip(list(label_encoder.classes_), range(0, len(label_encoder.classes_))))


def save_vectorizer(vect, model_dir: str) -> None:
    maybe_create_model_dir(model_dir)
    pickle.dump(vect, open(os.path.join(model_dir, 'vectorizer.pickle'), 'wb'))


def save_model(model, model_dir: str) -> None:
    maybe_create_model_dir(model_dir)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path, compress=1)


def save_label_encoder(label_encoder, model_dir: str) -> None:
    maybe_create_model_dir(model_dir)
    with open(os.path.join(model_dir, 'label_encoder.json'), 'w+') as out_file:
        params = get_classes_dictionary(label_encoder)
        json.dump(params, out_file, sort_keys=True)
