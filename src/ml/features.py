import csv
import os

import numpy as np
import pandas as pd

from ml.modelsaving import get_classes_dictionary, maybe_create_model_dir


def save_top_k_features(vectorizer,
                        clf,
                        model_dir,
                        label_encoder,
                        num_features=10):
    feature_names = vectorizer.get_feature_names()
    class_dictionary = get_classes_dictionary(label_encoder)
    rows = []
    for class_label, index in class_dictionary.items():
        feature_array = clf.coef_[index]
        if not isinstance(feature_array, np.ndarray):
            feature_array = feature_array.toarray()[0]
        top_k = np.argsort(feature_array)[-num_features:]
        rows.append([class_label, [feature_names[j].replace('_', ' ') for j in top_k]])
    maybe_create_model_dir(model_dir)
    df_output = pd.DataFrame(data=rows, columns=['Label', 'Features'])
    df_output.to_csv(os.path.join(model_dir, 'top_k_features.csv'),
                     sep=',',
                     encoding='utf-8',
                     index=False,
                     quoting=csv.QUOTE_ALL)
