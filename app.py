from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# APP SETUP
# ===============================
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# LOAD MODEL & ENCODERS (SAFE)
# ===============================
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
label_encoders = pickle.load(open(ENCODER_PATH, "rb"))

MODEL_FEATURES = [
    "Name_x",
    "State",
    "Type",
    "BestTimeToVisit",
    "Preferences",
    "Gender",
    "NumberOfAdults",
    "NumberOfChildren",
]

# ===============================
# LOAD DATASETS
# ===============================
DATA_DIR = os.path.join(BASE_DIR, "data")

destinations_df = pd.read_csv(
    os.path.join(DATA_DIR, "2000_Destinations_with_Popularity_and_BestTime.csv")
)

userhistory_df = pd.read_csv(
    os.path.join(DATA_DIR, "FINAL_2000_UserHistory_CLEANED.csv")
)

# ===============================
# CLEAN COLUMN NAMES
# ===============================
destinations_df.columns = destinations_df.columns.str.strip()
userhistory_df.columns = userhistory_df.columns.str.strip()

# ===============================
# COLLABORATIVE FILTERING SETUP
# ===============================
user_item_matrix = userhistory_df.pivot_table(
    index="UserID",
    columns="DestinationID",
    values="ExperienceRating",
    aggfunc="mean",
).fillna(0)

user_similarity = cosine_similarity(user_item_matrix.values.astype("float32"))

# ===============================
# SAFE ENCODER
# ===============================
def safe_encode(feature, value):
    """
    Encode categorical features safely.
    If unseen value appears, fallback to first known class.
    """
    if feature in label_encoders:
        encoder = label_encoders[feature]
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            return encoder.transform([encoder.classes_[0]])[0]
    return value


# ===============================
# RECOMMENDATION LOGIC
# ===============================
def collaborative_recommend(user_id, top_n=5):
    """
    User-based collaborative filtering
    """
    if user_id not in user_item_matrix.index:
        # New user → random popular destinations
        return destinations_df.sample(top_n)[
            ["Name", "State", "Type", "Popularity", "BestTimeToVisit"]
        ]

    user_idx = list(user_item_matrix.index).index(user_id)
    sim_scores = user_similarity[user_idx]

    similar_users = np.argsort(sim_scores)[::-1][1:6]
    avg_ratings = user_item_matrix.iloc[similar_users].mean(axis=0)

    top_dest_ids = avg_ratings.sort_values(ascending=False).head(top_n).index

    return destinations_df[
        destinations_df["DestinationID"].isin(top_dest_ids)
    ][["Name", "State", "Type", "Popularity", "BestTimeToVisit"]]


def predict_popularity(user_input):
    """
    Predict popularity using ML model
    """
    encoded_input = {
        feature: safe_encode(feature, user_input[feature])
        for feature in MODEL_FEATURES
    }

    input_df = pd.DataFrame([encoded_input])
    return float(model.predict(input_df)[0])


# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommendation")
def recommendation():
    return render_template(
        "recommendation.html",
        recommended_destinations=None,
        predicted_popularity=None,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = int(request.form["user_id"])
    except ValueError:
        user_id = -1  # fallback for safety

    preference = request.form["preferences"]

    user_input = {
        "Name_x": request.form["name"],
        "State": request.form["state"],
        "Type": preference,
        "BestTimeToVisit": request.form["best_time"],
        "Preferences": preference,
        "Gender": request.form["gender"],
        "NumberOfAdults": int(request.form["adults"]),
        "NumberOfChildren": int(request.form["children"]),
    }

    recommended_destinations = collaborative_recommend(user_id)
    predicted_popularity = predict_popularity(user_input)

    return render_template(
        "recommendation.html",
        recommended_destinations=recommended_destinations,
        predicted_popularity=predicted_popularity,
    )


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
