from sklearn.metrics import precision_recall_fscore_support

def evaluate_model(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(y_pred)
    print(f"Precision: {precision:.8f}")
    print(f"Recall: {recall:.8f}")
    print(f"F1-score: {f1:.8f}")
    #evaluate