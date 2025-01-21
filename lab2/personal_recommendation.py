import pandas as pd
import pickle
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

nltk.download("stopwords")
nltk.download("punkt")

def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    valid_books = ratings["ISBN"].value_counts()[lambda x: x > 1].index
    valid_users = ratings["User-ID"].value_counts()[lambda x: x > 1].index
    ratings = ratings[ratings["ISBN"].isin(valid_books) & ratings["User-ID"].isin(valid_users)]
    
    avg_rating_per_book = ratings.groupby("ISBN")["Book-Rating"].transform("mean")
    rating_counts_per_book = ratings.groupby("ISBN")["Book-Rating"].transform("count")
    ratings["Book-Rating"] = avg_rating_per_book
    ratings["Rating-Count"] = rating_counts_per_book

    ratings.dropna(inplace=True)
    
    return ratings

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
    df = df[df['Year-Of-Publication'] <= 2025][df['Year-Of-Publication'] > 1]
    return df

def get_user_with_most_zeros(ratings: pd.DataFrame) -> int:
    zero_ratings = ratings[ratings['Book-Rating'] == 0]
    return zero_ratings['User-ID'].value_counts().idxmax()

def predict_with_svd(user_id: int, svd_model: SVD, ratings: pd.DataFrame, threshold: int = 8):
    unseen_books = ratings[ratings['User-ID'] == user_id]['ISBN'].unique()
    predictions = {
        book: svd_model.predict(user_id, book).est
        for book in unseen_books
    }
    high_rated_books = [book for book, pred in predictions.items() if pred >= threshold]
    return high_rated_books

def preprocess_all_books(high_rated_books: pd.DataFrame, tfidf: TfidfVectorizer, scaler: StandardScaler):
    def title_preprocessing(text: str) -> str:
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    processed_titles = [title_preprocessing(title) for title in high_rated_books["Book-Title"]]
    title_vectors = tfidf.transform(processed_titles).toarray()

    author_codes = pd.Series(high_rated_books["Book-Author"]).astype("category").cat.codes
    publisher_codes = pd.Series(high_rated_books["Publisher"]).astype("category").cat.codes

    features = pd.DataFrame({
        "Book-Author": author_codes,
        "Publisher": publisher_codes,
        "Year-Of-Publication": high_rated_books["Year-Of-Publication"],
        "Rating-Count": high_rated_books["Rating-Count"]
    })

    for i in range(title_vectors.shape[1]):
        features[f"title_{i}"] = title_vectors[:, i]

    scaled_features = scaler.transform(features)
    return scaled_features

def predict_all_books(model, processed_data, high_rated_books):
    predictions = model.predict(processed_data)
    high_rated_books["LinReg-Prediction"] = predictions
    return high_rated_books.sort_values("LinReg-Prediction", ascending=False)

if __name__ == "__main__":
    books = pd.read_csv('./Books.csv')
    ratings = pd.read_csv('./Ratings.csv')

    ratings = preprocess_ratings(ratings)
    books = books_preprocessing(books)

    user_id = get_user_with_most_zeros(ratings)
    print(f"Пользователь с наибольшим количеством нулевых оценок: {user_id}")

    with open("svd.pkl", "rb") as file:
        svd_model = pickle.load(file)
    with open("linreg.pkl", "rb") as file:
        linreg_model = pickle.load(file)

    high_rated_isbns = predict_with_svd(user_id, svd_model, ratings)
    high_rated_books = books[books['ISBN'].isin(high_rated_isbns)]
    print(f"Книг, рекомендованных с помощью SVD (оценка >= 8): {len(high_rated_books)}")

    with open("tfidf.pkl", "rb") as file:
        tfidf_vectorizer = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    rating_counts = ratings.groupby('ISBN')['Book-Rating'].count()
    high_rated_books['Rating-Count'] = high_rated_books['ISBN'].map(rating_counts).fillna(0).astype(int)

    processed_data = preprocess_all_books(high_rated_books, tfidf_vectorizer, scaler)

    recommendations = predict_all_books(linreg_model, processed_data, high_rated_books)

    print("Топ-10 рекомендаций:")
    print("=" * 50)
    for idx, row in recommendations[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'LinReg-Prediction']].head(10).iterrows():
        print(f"Книга: «{row['Book-Title']}»")
        print(f"Автор: {row['Book-Author']}")
        print(f"Год выпуска: {int(row['Year-Of-Publication'])}")
        print(f"Предполагаемая оценка: {row['LinReg-Prediction']:.2f}")
        print("-" * 50)

    with open('recommendations.txt', 'w', encoding='utf-8') as file:
        file.write("Рекомендации книг на основе предсказаний модели LinReg\n")
        file.write("=" * 50 + "\n\n")
        
        for idx, row in recommendations[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'LinReg-Prediction']].iterrows():
            file.write(f"Книга: «{row['Book-Title']}»\n")
            file.write(f"Автор: {row['Book-Author']}\n")
            file.write(f"Год выпуска: {int(row['Year-Of-Publication'])}\n")
            file.write(f"Предполагаемая оценка: {row['LinReg-Prediction']:.2f}\n")
            file.write("-" * 50 + "\n")

'''
Рекомендации книг на основе предсказаний модели LinReg
==================================================

Книга: «Harry Potter and the Chamber of Secrets (Book 2)»
Автор: J. K. Rowling
Год выпуска: 2000
Предполагаемая оценка: 8.99
--------------------------------------------------
Книга: «The Magician's Nephew (rack) (Narnia)»
Автор: C. S. Lewis
Год выпуска: 2002
Предполагаемая оценка: 8.97
--------------------------------------------------
Книга: «A Prayer for Owen Meany»
Автор: John Irving
Год выпуска: 1990
Предполагаемая оценка: 8.84
--------------------------------------------------
Книга: «The Lion, the Witch and the Wardrobe (Full-Color Collector's Edition)»
Автор: C. S. Lewis
Год выпуска: 2000
Предполагаемая оценка: 8.78
--------------------------------------------------
Книга: «The Color Purple»
Автор: Alice Walker
Год выпуска: 1985
Предполагаемая оценка: 8.75
--------------------------------------------------
Книга: «Charlotte's Web (Trophy Newbery)»
Автор: E. B. White
Год выпуска: 1974
Предполагаемая оценка: 8.63
--------------------------------------------------
Книга: «Far Side Gallery 2»
Автор: Gary Larson
Год выпуска: 2003
Предполагаемая оценка: 8.54
--------------------------------------------------
Книга: «The Giver (21st Century Reference)»
Автор: LOIS LOWRY
Год выпуска: 1994
Предполагаемая оценка: 8.33
--------------------------------------------------
Книга: «The Phantom Tollbooth»
Автор: Norton Juster
Год выпуска: 1993
Предполагаемая оценка: 8.28
--------------------------------------------------
Книга: «A Wrinkle In Time»
Автор: MADELEINE L'ENGLE
Год выпуска: 1998
Предполагаемая оценка: 8.05
--------------------------------------------------
Книга: «A Wrinkle in Time»
Автор: Madeleine L'Engle
Год выпуска: 1976
Предполагаемая оценка: 8.03
--------------------------------------------------
Книга: «Snow White and the Seven Dwarfs»
Автор: Little Golden Staff
Год выпуска: 1994
Предполагаемая оценка: 7.96
--------------------------------------------------
Книга: «The Hot Zone»
Автор: Richard Preston
Год выпуска: 1994
Предполагаемая оценка: 7.81
--------------------------------------------------
Книга: «Goodnight Moon Board Book»
Автор: Margaret Wise Brown
Год выпуска: 1991
Предполагаемая оценка: 7.81
--------------------------------------------------
Книга: «The Black Cauldron (Chronicles of Prydain (Paperback))»
Автор: LLOYD ALEXANDER
Год выпуска: 1985
Предполагаемая оценка: 7.80
--------------------------------------------------
Книга: «Message from Nam»
Автор: Danielle Steel
Год выпуска: 1991
Предполагаемая оценка: 7.74
--------------------------------------------------
Книга: «October Sky: A Memoir»
Автор: Homer Hickam
Год выпуска: 1999
Предполагаемая оценка: 7.73
--------------------------------------------------
Книга: «Bridge to Terabithia»
Автор: Katherine Paterson
Год выпуска: 1987
Предполагаемая оценка: 7.68
--------------------------------------------------
Книга: «Lonesome Dove»
Автор: Larry McMurtry
Год выпуска: 1988
Предполагаемая оценка: 7.68
--------------------------------------------------
Книга: «Love You Forever»
Автор: Robert N. Munsch
Год выпуска: 1986
Предполагаемая оценка: 7.62
--------------------------------------------------
Книга: «Yukon Ho!»
Автор: Bill Watterson
Год выпуска: 1989
Предполагаемая оценка: 7.61
--------------------------------------------------
Книга: «She Said Yes : The Unlikely Martyrdom of Cassie Bernall»
Автор: Misty Bernall
Год выпуска: 2000
Предполагаемая оценка: 7.61
--------------------------------------------------
Книга: «Frindle»
Автор: Andrew Clements
Год выпуска: 1998
Предполагаемая оценка: 7.59
--------------------------------------------------

<<<<<<< Updated upstream
'''
=======
'''
>>>>>>> Stashed changes
