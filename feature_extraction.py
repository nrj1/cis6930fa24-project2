import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

nltk.download()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_features(data):
    # Extract n-grams
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(data['context'])
    
    # Extract redaction length
    X_redaction_length = data['context'].apply(lambda x: x.count('█'))
    
    # Extract POS tags and NER
    X_pos_ner = data['context'].apply(extract_pos_ner)
    
    # Combine features
    X_combined = pd.concat([pd.DataFrame(X.toarray()), X_redaction_length, X_pos_ner], axis=1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['name'])
    
    return X_combined, y

def extract_pos_ner(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    ner_tags = nltk.ne_chunk(pos_tags)
    
    # Extract relevant features from POS and NER tags
    # This is a simplified example and can be expanded
    return pd.Series({
        'num_nouns': len([word for word, pos in pos_tags if pos.startswith('NN')]),
        'num_verbs': len([word for word, pos in pos_tags if pos.startswith('VB')]),
        'num_ner': len([chunk for chunk in ner_tags if hasattr(chunk, 'label')])
    })
#extraction