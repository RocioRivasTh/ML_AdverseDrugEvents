
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
import pandas as pd

def smiles_to_vector(smiles_series, ngram_range=(2, 4), max_features=1000):
    """Convert SMILES strings into character-level n-gram vectors."""
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(smiles_series)
    return X, vectorizer

def evaluate_multilabel(y_true, y_pred):
    """Print multilabel classification report with micro and macro F1."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Micro F1 Score:", f1_score(y_true, y_pred, average='micro'))
    print("Macro F1 Score:", f1_score(y_true, y_pred, average='macro'))

def attach_predictions(X_index, predictions, y_columns, compound_names=None):
    """Create a DataFrame combining predictions and compound names."""
    pred_df = pd.DataFrame(predictions, columns=y_columns)
    if compound_names is not None:
        pred_df.insert(0, "Compound_Name", compound_names[X_index])
    return pred_df
