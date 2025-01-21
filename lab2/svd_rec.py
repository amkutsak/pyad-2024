import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings[ratings["Book-Rating"] != 0]

    valid_users = ratings["User-ID"].value_counts()[lambda x: x >= 1].index
    valid_books = ratings["ISBN"].value_counts()[lambda x: x >= 1].index
    ratings = ratings[ratings["User-ID"].isin(valid_users) & ratings["ISBN"].isin(valid_books)]

    ratings = ratings.drop_duplicates()
    ratings.dropna(inplace=True)
    return ratings

def modeling(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings[["User-ID", "ISBN", "Book-Rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.1, random_state=42)
    param_grid = {
        'n_factors': [20, 30, 40, 50, 60],
        'reg_all': [0.02, 0.05, 0.1, 0.15],
        'lr_all': [0.002, 0.005, 0.01],
        'n_epochs': [20, 30, 50]
    }

    grid_search = GridSearchCV(SVD, param_grid, measures=['mae'], cv=3, n_jobs=-1)
    grid_search.fit(data)

    best_params = grid_search.best_params['mae']
    print(f"Best Parameters: {best_params}")
    svd = SVD(**best_params)

    svd.fit(trainset)
    predictions = svd.test(testset)
    mae = accuracy.mae(predictions)
    print(f"Test MAE: {mae}")

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)

if __name__ == "__main__":
    ratings = pd.read_csv("./Ratings.csv")
    processed_ratings = preprocess_ratings(ratings)
    modeling(processed_ratings)
