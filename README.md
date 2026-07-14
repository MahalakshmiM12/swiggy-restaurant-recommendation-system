# Swiggy's Restaurant Recommendation System using Streamlit

A content-based restaurant recommendation system built on Swiggy's restaurant dataset,
using clustering and cosine similarity, with a Streamlit app for interactive use.

## Problem Statement

Recommend restaurants to users based on their preferred city, cuisine, rating, and budget,
using a dataset of ~148,500 real Swiggy restaurant listings.

## Business Use Cases

- **Personalized Recommendations:** Help users discover restaurants based on their preferences.
- **Improved Customer Experience:** Provide tailored suggestions to enhance decision-making.
- **Market Insights:** Understand customer preferences and behaviors for targeted marketing.
- **Operational Efficiency:** Enable businesses to optimize their offerings based on popular preferences.

## Approach

1. **Data Cleaning** — removed unusable rows, fixed the `'--'` rating sentinel, parsed the
   bucketed `rating_count` and `₹`-prefixed `cost` columns, filtered promotional text out of
   `cuisine`, and reasonably imputed remaining missing values city-by-city (median/mode).
2. **Encoding** — one-hot encoded `city` (top 150 + "Other") and multi-hot encoded `cuisine`;
   standardized `rating`, `rating_count`, `cost`.
3. **Clustering** — MiniBatchKMeans (k=8) segments restaurants into market groups.
4. **Recommendation Engine** — cosine similarity between user preferences and each restaurant's
   encoded profile, filtered by city/budget/rating, ranked, and mapped back to readable data.

## Tech Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/MahalakshmiM12/swiggy-restaurant-recommendation-system.git
```

### 2. Navigate to the project directory

```
cd swiggy-restaurant-recommendation-system
```

### 3. Create a virtual environment

```
python -m venv .venv
```

### 4. Activate the virtual environment

```
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

### 5. Install dependencies

```
pip install -r requirements.txt
```

### 6. Add the dataset

Place `swiggy.csv` in the same folder as the notebook (it is not included in the repo —
add your own copy).

### 7. Run the notebook (data cleaning → encoding → clustering → recommendations)

Open `swiggy_restaurant_recommendation_system.ipynb` in Jupyter and run all cells:

```
jupyter notebook swiggy_restaurant_recommendation_system.ipynb
```

This generates `cleaned_data.csv`, `encoded_data.csv`, `encoder.pkl`, and `kmeans_model.pkl`,
which the Streamlit app depends on.

### 8. Run the Streamlit app

```
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).
