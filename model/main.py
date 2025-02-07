import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    # Split the data into X and y
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit the model
    model = LogisticRegression()
    model.fit(X_train, y_train)  
    # Test The Model

    y_pred = model.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, y_pred) * 100)

    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data():
    # Import the dataset
    data = pd.read_csv('data/data.csv')

    # Drop unnecessary columns
    data = data.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')  # Fixed: Avoid redundant drop

    # Encode the diagnosis variable
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open ('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open ('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
