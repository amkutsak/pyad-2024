import pickle
import re
import nltk
import pandas as pd
import sklearn
import string
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')
<<<<<<< Updated upstream
=======

import pickle

def save_preprocessors(tfidf: TfidfVectorizer, scaler: StandardScaler):
    with open("tfidf.pkl", "wb") as tfidf_file:
        pickle.dump(tfidf, tfidf_file)

    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
>>>>>>> Stashed changes

import pickle

def save_preprocessors(tfidf: TfidfVectorizer, scaler: StandardScaler):
    with open("tfidf.pkl", "wb") as tfidf_file:
        pickle.dump(tfidf, tfidf_file)

    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    data = {
        "index": [209538, 220731, 221678],
        "ISBN": ["078946697X", "2070426769", "0789466953"],
        "Book-Title": [
            'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)',
            "Peuple du ciel, suivi de 'Les Bergers",
            'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)',
        ],
        "Book-Author": [
            "Michael Teitelbaum",
            "Jean-Marie Gustave Le Clézio",
            'James Buckley'
        ],
        "Year-Of-Publication": [2000, 2003, 2000],
        "Publisher": ["DK Publishing Inc", "Gallimard", "DK Publishing Inc"],
        "Image-URL-S": [
            "http://images.amazon.com/images/P/078946697X.01.THUMBZZZ.jpg",
            "http://images.amazon.com/images/P/2070426769.01.THUMBZZZ.jpg",
            "http://images.amazon.com/images/P/0789466953.01.THUMBZZZ.jpg",
        ],
        "Image-URL-M": [
            "http://images.amazon.com/images/P/078946697X.01.MZZZZZZZ.jpg",
            "http://images.amazon.com/images/P/2070426769.01.MZZZZZZZ.jpg",
            "http://images.amazon.com/images/P/0789466953.01.MZZZZZZZ.jpg",
        ],
        "Image-URL-L": [
            "http://images.amazon.com/images/P/078946697X.01.LZZZZZZZ.jpg",
            "http://images.amazon.com/images/P/2070426769.01.LZZZZZZZ.jpg",
            "http://images.amazon.com/images/P/0789466953.01.LZZZZZZZ.jpg",
        ],
    }
    new_data = pd.DataFrame(data)
    new_data.set_index('index', inplace=True)
    df.update(new_data)
    df.loc[df["ISBN"] == "0751352497", "Book-Author"] = "Dorling Kindersley"
    df.loc[df["ISBN"] == "193169656X", "Publisher"] = "Phobos Books"
    df.loc[df["ISBN"] == "1931696993", "Publisher"] = "Phobos Books"
    df.loc[df["ISBN"] == "9627982032", "Book-Author"] = "Credit Suisse"
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df = df[df['Year-Of-Publication'] <= 2025]
    return df

def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings[ratings["Book-Rating"] != 0]

    valid_books = ratings["ISBN"].value_counts()[lambda x: x > 1].index
    valid_users = ratings["User-ID"].value_counts()[lambda x: x > 1].index
    ratings = ratings[ratings["ISBN"].isin(valid_books) & ratings["User-ID"].isin(valid_users)]
    
    avg_rating_per_book = ratings.groupby("ISBN")["Book-Rating"].transform("mean")
    rating_counts_per_book = ratings.groupby("ISBN")["Book-Rating"].transform("count")
    ratings["Book-Rating"] = avg_rating_per_book
    ratings["Rating-Count"] = rating_counts_per_book
    
    return ratings

def title_preprocessing(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    processed_text = " ".join(tokens)
    
    return processed_text
<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    books["Processed-Title"] = books["Book-Title"].apply(title_preprocessing)

    books["Book-Author"] = books["Book-Author"].astype("category").cat.codes
    books["Publisher"] = books["Publisher"].astype("category").cat.codes

    data = pd.merge(ratings, books, on="ISBN")
    
    features = ["Processed-Title", "Book-Author", "Publisher", "Year-Of-Publication", "Rating-Count"]
    target = "Book-Rating"
    
    tfidf = TfidfVectorizer(max_features=1000)
    title_vectors = tfidf.fit_transform(data["Processed-Title"]).toarray()
    title_df = pd.DataFrame(title_vectors, columns=[f"title_{i}" for i in range(1000)])
    
    X = pd.concat([data[features].drop(columns=["Processed-Title"]), title_df], axis=1)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linreg = SGDRegressor(random_state=42)
    linreg.fit(X_train_scaled, y_train)

    y_pred = linreg.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")

    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)
    save_preprocessors(tfidf, scaler)

if __name__ == "__main__":
    books = pd.read_csv('./Books.csv')
    ratings = pd.read_csv('./Ratings.csv')
    books = books_preprocessing(books)
    ratings = preprocess_ratings(ratings)
    modeling(books, ratings)
<<<<<<< Updated upstream
=======
    
>>>>>>> Stashed changes
