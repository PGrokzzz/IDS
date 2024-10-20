# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# def train_model():
#     # Load the dataset

#     # Preprocess the data...
#     if 'Malicious' not in data.columns:
#         data['Malicious'] = np.random.randint(0, 2, size=len(data))

#     label_encoders = {
#         'Source': LabelEncoder(),
#         'Destination': LabelEncoder(),
#         'Info': LabelEncoder(),
#         'Protocol': LabelEncoder()
#     }

#     # Fit the encoders
#     for col, encoder in label_encoders.items():
#         data[col] = encoder.fit_transform(data[col])

#     X = data.drop('Malicious', axis=1)
#     y = data['Malicious']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
#     clf.fit(X_train, y_train)

#     # Return the trained model and label encoders
#     return clf, label_encoders
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define encoders globally so they can be reused
label_encoders = {
    'Source': LabelEncoder(),
    'Destination': LabelEncoder(),
    'Info': LabelEncoder(),
    'Protocol': LabelEncoder()
}

def preprocess_input_data(input_data):
    """Preprocess input data before making predictions."""
    # Assuming input_data is a list of [Source, Destination, Protocol, Length, Info, Flag]
    
    # Apply the same label encoding that was used during training
    input_data[0] = label_encoders['Source'].transform([input_data[0]])[0]  # Encode Source
    input_data[1] = label_encoders['Destination'].transform([input_data[1]])[0]  # Encode Destination
    input_data[2] = label_encoders['Protocol'].transform([input_data[2]])[0]  # Encode Protocol
    input_data[4] = label_encoders['Info'].transform([input_data[4]])[0]  # Encode Info
    
    # Return the preprocessed data
    return input_data


def train_model():
    # Load the dataset
    data = pd.read_csv('sampled_1000_values.csv')

    # Preprocess the data...
    if 'Malicious' not in data.columns:
        data['Malicious'] = np.random.randint(0, 2, size=len(data))

    for col, encoder in label_encoders.items():
        data[col] = encoder.fit_transform(data[col])

    X = data.drop('Malicious', axis=1)
    y = data['Malicious']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    clf.fit(X_train, y_train)

    return clf  # Return the trained model
