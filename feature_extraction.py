import nltk
import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

def extract_features_train(data):
   
    
    # Extract n-grams
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform(data['context'])
        
    # Extract redaction length
    X_redaction_length = data['context'].apply(lambda x: x.count('█'))
    
    # Extract POS tags and NER
    X_pos_ner = data['context'].apply(extract_pos_ner)
    
    block_lengths_array=np.array(X_redaction_length).reshape(-1,1)

    X_combined=np.hstack([X.toarray(),block_lengths_array])
    
    # Combine features
    #X_combined = pd.concat([X_redaction_length, X_pos_ner], axis=1)
    
    # Encode labels (only for non-test data)
    #if not is_test:
        #label_encoder = LabelEncoder()
        #y = label_encoder.fit_transform(data['name'])
    return X_combined, data['name'],vectorizer
    #else:
        # Return features only for test data
    #    return X_combined, None
    
def extract_features_val_test(data,vectorizer, is_test=False):
    # Extract n-grams
    X = vectorizer.transform(data['context'])
        
    # Extract redaction length
    X_redaction_length = data['context'].apply(lambda x: x.count('█'))
    
    # Extract POS tags and NER
    X_pos_ner = data['context'].apply(extract_pos_ner)
    
    block_lengths_array=np.array(X_redaction_length).reshape(-1,1)

    X_combined=np.hstack([X.toarray(),block_lengths_array])
    
    # Combine features
    #X_combined = pd.concat([X_redaction_length, X_pos_ner], axis=1)
    
    # Encode labels (only for non-test data)
    if not is_test:
        #label_encoder = LabelEncoder()
        #y = label_encoder.fit_transform(data['name'])
        return X_combined, data['name']
    else:
        # Return features only for test data
        return X_combined, None




def extract_pos_ner(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    ner_tags = nltk.ne_chunk(pos_tags)
    
    # Extract relevant features from POS and NER tags
    return pd.Series({
        'num_nouns': len([word for word, pos in pos_tags if pos.startswith('NN')]),
        'num_verbs': len([word for word, pos in pos_tags if pos.startswith('VB')]),
        'num_ner': len([chunk for chunk in ner_tags if hasattr(chunk, 'label')])
    })
