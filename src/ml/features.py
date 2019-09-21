import os
import matplotlib.pyplot as plt
import numpy as np


def plot_top_k_features(vectorizer,
                        clf,
                        model_path,
                        num_features=10):
    feature_names = np.array(vectorizer.get_feature_names())

    coefficients = clf.coef_[0].toarray()[0]
    top_positive_coefficients = np.argsort(coefficients)[-num_features:]
    top_negative_coefficients = np.argsort(coefficients)[:num_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # Create features plot
    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    colors = ['red' if coef < 0 else 'blue' for coef in coefficients[top_coefficients]]
    plt.bar(np.arange(2 * num_features), coefficients[top_coefficients], color=colors)
    plt.xticks(np.arange(0, 2 * num_features), feature_names[top_coefficients], rotation=80, ha='right')
    plt.title('Important Features - For Non-Marketing', fontsize=18)
    plt.savefig(os.path.join(model_path, 'important_features.png'))
