import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Melbourne Airbnb Dashboard", layout="centered")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("cleansed_listings_dec18.csv")

df = load_data()
st.title("🏡 Melbourne Airbnb - Data Exploration & Explainable ML")

st.write("This Streamlit app was developed as part of a project for Country Financial. Because we didn’t have access to CF’s proprietary claims and call-summary data, we used a publicly available Melbourne Airbnb dataset that closely mimics the structure CF would work with, containing both unstructured text and tabular customer information. The app offers an interactive interface for exploring listings and employs explainability techniques to show not only what the model predicts but why. Country Financial will adapt this framework to predict and explain subrogation decisions instead of prices, making AI-driven recommendations transparent and actionable.")
# Sidebar filters
st.sidebar.header("🔍 Filter Listings")
selected_room_type = st.sidebar.multiselect("Room Type", df['room_type'].unique(), default=df['room_type'].unique())
filtered_df = df[df['room_type'].isin(selected_room_type)]

# Data Overview
st.header("🔢 Dataset Preview")
st.dataframe(filtered_df.head())

with st.expander("📊 Dataset Summary", expanded=False):
    # --- Numerical features ---
    st.subheader("Numerical Features")
    num_cols = filtered_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if num_cols:
        # summary table
        st.write(filtered_df[num_cols].describe())
        # user selects one numeric column
        selected_num = st.selectbox("Choose a numeric feature to plot", num_cols)
        # plot its histogram
        fig, ax = plt.subplots()
        ax.hist(filtered_df[selected_num].dropna(), bins=20)
        ax.set_title(f"Distribution of {selected_num}")
        ax.set_xlabel(selected_num)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.write("No numeric columns found.")

    # --- Text features ---
    st.subheader("Text Features")
    text_cols = ["name", "summary", "description", "neighborhood_overview"]
    text_cols = [c for c in text_cols if c in filtered_df.columns]
    if text_cols:
        # compute & show length stats
        stats = []
        for col in text_cols:
            lengths = filtered_df[col].dropna().astype(str).apply(len)
            stats.append({
                "column": col,
                "count": lengths.count(),
                "mean_length": lengths.mean(),
                "median_length": lengths.median(),
                "min_length": lengths.min(),
                "max_length": lengths.max()
            })
        st.write(pd.DataFrame(stats).set_index("column"))

        # user selects one text column
        selected_text = st.selectbox("Choose a text column to plot length distribution", text_cols)
        lengths = filtered_df[selected_text].dropna().astype(str).apply(len)
        # plot its length histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(lengths, bins=20)
        ax2.set_title(f"Length Distribution for {selected_text}")
        ax2.set_xlabel("Character Count")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
    else:
        st.write("No text columns found.")

# Train Deep Learning Model
@st.cache_resource
def train_deep_learning_model(df):
    tabular_features = [
        'host_is_superhost', 'latitude', 'longitude',
        'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'number_of_reviews', 'review_scores_rating',
        'calculated_host_listings_count'
    ]

    text_cols = ['name', 'summary', 'description', 'neighborhood_overview']

    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
    tabular_df = df[tabular_features]
    imputer = SimpleImputer(strategy='mean')
    tabular_df_imputed = pd.DataFrame(imputer.fit_transform(tabular_df), columns=tabular_features)

    text_data = df[text_cols].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=300)
    text_features = tfidf.fit_transform(text_data)

    X_combined = hstack([tabular_df_imputed.values, text_features])
    y = (df['price'] > 100).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_combined.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        X_train.toarray(), y_train,
        validation_split=0.2,
        epochs=5,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    return model, imputer, tfidf, X_train, X_test, y_train


with st.spinner("Training deep learning model..."):
    model_dl, imputer_dl, tfidf_dl, X_train_dl, X_test_dl, y_train_dl = train_deep_learning_model(df)
st.success("Deep learning model trained and ready!")

# 🔎 Search by ID and display tabular + text data
st.header("Individual observation explainability")
st.subheader("🔎 Search Listing by ID")
listing_ids = df['id'].unique()
search_id = st.text_input("Enter Listing ID")

selected_obs_combined = None
obs_tabular_imputed = None
shap_force_plot_ready = False

if search_id != "":
    try:
        search_id = int(search_id)
        if search_id in listing_ids:
            st.success(f"Listing ID {search_id} found!")

            selected_row = df[df['id'] == search_id].iloc[0]

            tabular_features = [
                'host_is_superhost', 'latitude', 'longitude',
                'accommodates', 'bathrooms', 'bedrooms', 'beds',
                'minimum_nights', 'number_of_reviews', 'review_scores_rating',
                'calculated_host_listings_count'
            ]

            text_cols = ['name', 'summary', 'description', 'neighborhood_overview']

            st.subheader("📋 Tabular Features")
            st.dataframe(pd.DataFrame(selected_row[tabular_features]).transpose())

            st.subheader("📝 Text Information")
            for col in text_cols:
                if pd.notnull(selected_row[col]):
                    st.markdown(f"**{col.replace('_', ' ').title()}:** {selected_row[col]}")
                else:
                    st.markdown(f"**{col.replace('_', ' ').title()}:** _No information available_")

            obs = df[df['id'] == search_id].copy().reset_index(drop=True)
            obs['host_is_superhost'] = obs['host_is_superhost'].map({'t': 1, 'f': 0})
            obs_tabular = obs[tabular_features]
            obs_tabular_imputed = pd.DataFrame(imputer_dl.transform(obs_tabular), columns=tabular_features)

            obs_text = obs[text_cols].fillna('').agg(' '.join, axis=1)
            obs_text_tfidf = tfidf_dl.transform(obs_text)

            selected_obs_combined = hstack([obs_tabular_imputed.values, obs_text_tfidf])

            pred_prob = model_dl.predict(selected_obs_combined.toarray())[0][0]
            pred_class = int(pred_prob > 0.5)
            label = "Expensive (Price > $100)" if pred_class else "Affordable (≤ $100)"

            st.metric(label="🔮 Prediction", value=label)
            st.caption(f"Probability of being expensive: {pred_prob:.2f}")

            shap_force_plot_ready = True

            # Show force plot here inside prediction block
            import shap
            import numpy as np
            from streamlit.components.v1 import html

            X_sample = X_train_dl.toarray()[:100]
            explainer = shap.DeepExplainer(model_dl, X_sample)

            shap_values_obs_full = explainer.shap_values(selected_obs_combined.toarray())

            if isinstance(shap_values_obs_full, list):
                shap_values_obs_full = shap_values_obs_full[0][0]
                expected_value = float(explainer.expected_value[0])
            else:
                shap_values_obs_full = shap_values_obs_full[0]
                expected_value = float(explainer.expected_value)

            tabular_part = np.array(shap_values_obs_full[:len(tabular_features)])
            text_part = np.array(shap_values_obs_full[len(tabular_features):])
            text_sum = float(text_part.sum())

            shap_values_obs_combined = np.concatenate([tabular_part.reshape(-1), np.array([text_sum])])
            combined_data = np.concatenate([obs_tabular_imputed.iloc[0].values.astype(float), np.array([0.0])])
            all_feature_names = tabular_features + ["ALL_TEXT_FEATURES"]

            explanation_force = shap.Explanation(
                values=shap_values_obs_combined,
                base_values=np.array([expected_value]),
                data=combined_data.reshape(-1),
                feature_names=all_feature_names
            )

            st.subheader("🧭 SHAP Force Plot (Selected Listing - Tabular + Summed Text)")
            _ = shap.plots.force(expected_value, shap_values_obs_combined, feature_names=all_feature_names, matplotlib=True, show = False)
            fig_force = plt.gcf()
            st.pyplot(fig_force)
            
        else:
            st.error("Listing ID not found in the dataset.")
    except ValueError:
        st.error("Please enter a valid numeric ID.")

try:
    import lime
    import lime.lime_text
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # new

    if search_id:
        try:
            search_id = int(search_id)
            if search_id not in listing_ids:
                st.error("Listing ID not found.")
            else:

                # Prepare row
                row = df[df['id'] == search_id].copy().reset_index(drop=True)
                row['host_is_superhost'] = row['host_is_superhost'].map({'t': 1, 'f': 0})

                tabular_features = [
                    'host_is_superhost', 'latitude', 'longitude',
                    'accommodates', 'bathrooms', 'bedrooms', 'beds',
                    'minimum_nights', 'number_of_reviews', 'review_scores_rating',
                    'calculated_host_listings_count'
                ]
                text_cols = ['name', 'summary', 'description', 'neighborhood_overview']

                X_tab_sample = row[tabular_features]
                text_sample = row[text_cols].fillna('').agg(' '.join, axis=1).iloc[0]

                # --- remove stopwords here ---
                text_sample = ' '.join([
                    w for w in text_sample.split()
                    if w.lower() not in ENGLISH_STOP_WORDS
                ])

                true_price = (row['price'] > 100).astype(int).iloc[0]

                # Define prediction function
                def predict_fn(texts):
                    text_tfidf = tfidf_dl.transform(texts)
                    text_dense = text_tfidf.toarray()
                    tab_repeat = np.repeat(X_tab_sample.values, len(texts), axis=0)
                    combined = np.hstack([tab_repeat, text_dense])
                    preds = model_dl.predict(combined)
                    return preds.reshape(-1, 1)

                # LIME Explainer
                lime_text_explainer = lime.lime_text.LimeTextExplainer(class_names=["Affordable", "Expensive"])
                lime_exp = lime_text_explainer.explain_instance(
                    text_sample,
                    predict_fn,
                    labels=[0],
                    num_features=10
                )

                # 🔍 Custom LIME bar chart
                st.markdown("### 🔍 Top Token Contributions (Bar Chart)")
                weights = lime_exp.as_list(label=0)
                features, values = zip(*weights)

                colors = ['orange' if v > 0 else 'blue' for v in values]
                fig2, ax = plt.subplots(figsize=(10, 6))
                ax.barh(features[::-1], values[::-1], color=colors[::-1])
                ax.set_xlabel("LIME Weight")
                ax.set_title("Top Words Driving the Prediction")
                st.pyplot(fig2)

                # Display full LIME HTML with a white background
                lime_html = lime_exp.as_html()
                custom_css = """
                <style>
                body, .lime, .table, .table th, .table td {
                    background-color: white !important;
                    color: black !important;
                }
                .highlight {
                    background-color: #1f77b4 !important;
                    color: white !important;
                }
                .highlight.positive {
                    background-color: #2ca02c !important;
                }
                .highlight.negative {
                    background-color: #d62728 !important;
                }
                </style>
                """
                patched_html = lime_html.replace("</head>", custom_css + "</head>")
                st.components.v1.html(patched_html, height=500, scrolling=True)

        except ValueError:
            st.error("Please enter a valid numeric ID.")
except Exception as e:
    st.error(f"Error running LIME explanation: {e}")

st.header("Overall Model Explainability")

# SHAP values for Deep Learning model
with st.expander("Tabular Features by SHAP Importance"):
    try:
        import shap
        import numpy as np

        X_sample = X_train_dl.toarray()[:100]
        explainer = shap.DeepExplainer(model_dl, X_sample)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values_np = shap_values[0].squeeze()
        else:
            shap_values_np = shap_values.squeeze()

        tabular_features = [
            'host_is_superhost', 'latitude', 'longitude',
            'accommodates', 'bathrooms', 'bedrooms', 'beds',
            'minimum_nights', 'number_of_reviews', 'review_scores_rating',
            'calculated_host_listings_count'
        ]

        tabular_shap_values = shap_values_np[:, :len(tabular_features)]
        tabular_data = X_sample[:, :len(tabular_features)]

        explanation_tabular = shap.Explanation(
            values=tabular_shap_values,
            data=tabular_data,
            feature_names=tabular_features
        )

        st.subheader("📈 SHAP Beeswarm Plot (Tabular Features Only)")
        fig_swarm, ax_swarm = plt.subplots(figsize=(8, 5))
        shap.plots.beeswarm(explanation_tabular, show=False)
        st.pyplot(fig_swarm)

    except Exception as e:
        raise e

# 🔍 Show SHAP text contribution per observation
with st.expander("SHAP Value of Text for Each Observation"):
    try:
        # Get SHAP values for 100 samples
        X_sample = X_train_dl.toarray()[:100]
        explainer = shap.DeepExplainer(model_dl, X_sample)
        shap_values = explainer.shap_values(X_sample)

        # Ensure we’re working with the right format
        if isinstance(shap_values, list):
            shap_values_np = shap_values[0]
        else:
            shap_values_np = shap_values

        # Text features are the last 300 columns (after tabular)
        text_shap_vals = shap_values_np[:, len(tabular_features):]

        # Sum SHAP values across all text features per row
        text_shap_sum_per_obs = text_shap_vals.sum(axis=1)

        # Build DataFrame
        text_shap_df = pd.DataFrame({
            "Observation": range(1, len(text_shap_sum_per_obs) + 1),
            "Text SHAP Value": text_shap_sum_per_obs.flatten()
        })

        # Display table
        st.dataframe(text_shap_df)

        # Optional: Display bar plot
        st.subheader("📉 Text SHAP Contribution per Observation")
        plt.figure(figsize=(10, 4))
        sns.barplot(data=text_shap_df, x="Observation", y="Text SHAP Value", palette="coolwarm")
        plt.xticks(rotation=90)
        plt.title("Net SHAP Value from Text Features")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error calculating text SHAP values: {e}")

with st.expander("SHAP Value of Each Word in Dataset"):
    try:
        import numpy as np
        from collections import Counter
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        import matplotlib.pyplot as plt

        # Define which text columns to combine
        text_cols = ['name', 'summary', 'description', 'neighborhood_overview']

        # 1. Compute net SHAP for each word in the TF-IDF vocabulary
        text_shap_vals = shap_values_np[:, len(tabular_features):]
        net_shap_per_word = text_shap_vals.sum(axis=0)
        tfidf_vocab = tfidf_dl.get_feature_names_out()

        # 2. Concatenate all text fields into one document per listing
        text_data = df[text_cols].fillna('').agg(' '.join, axis=1)


        # 3. Count actual word occurrences across all listings
        word_counts = Counter()
        for doc in text_data:
            tokens = doc.split()
            word_counts.update(tokens)
        true_freqs = [word_counts.get(word, 0) for word in tfidf_vocab]

        # 4. Build DataFrame of Word, SHAP, and actual Frequency
        word_stats_df = pd.DataFrame({
            'Word':            tfidf_vocab,
            'Net SHAP Value':  net_shap_per_word.flatten(),
            'Frequency':       true_freqs
        })

        # 5. Filter out common stopwords
        filtered_word_stats_df = word_stats_df[
            ~word_stats_df['Word'].isin(ENGLISH_STOP_WORDS)
        ]

        # 6. Select top 20 words by absolute SHAP impact
        top_words_df = (
            filtered_word_stats_df
            .reindex(filtered_word_stats_df['Net SHAP Value'].abs()
                     .sort_values(ascending=False)
                     .index)
            .head(20)
        )

        # 7. Display the table
        st.dataframe(top_words_df)

        # 8. Create overlaid horizontal bar chart
        fig, ax_shap = plt.subplots(figsize=(12, 7))
        ax_freq = ax_shap.twiny()

        y = np.arange(len(top_words_df))
        base_height = 0.5
        overlay_height = 0.25

        shap_vals = top_words_df['Net SHAP Value'].values
        freqs    = top_words_df['Frequency'].values

        # Scale frequencies to the same range as SHAP values
        freq_scale = np.max(np.abs(shap_vals)) / max(np.max(freqs), 1)
        scaled_freqs   = freqs * freq_scale
        mirrored_freqs = np.sign(shap_vals) * scaled_freqs

        shap_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in shap_vals]
        freq_colors = ['#a8e6a3' if v >= 0 else '#f4aaaa' for v in shap_vals]

        # Plot frequency bars (background)
        for i in range(len(y)):
            ax_shap.barh(
                y[i], mirrored_freqs[i],
                height=base_height,
                color=freq_colors[i],
                edgecolor='black',
                linewidth=1,
                zorder=1,
                label='Frequency (Scaled)' if i == 0 else None
            )

        # Overlay SHAP bars (foreground)
        ax_shap.barh(
            y, shap_vals,
            height=overlay_height,
            color=shap_colors,
            zorder=2,
            label='Net SHAP Value'
        )

        # Formatting
        ax_shap.set_yticks(y)
        ax_shap.set_yticklabels(top_words_df['Word'])
        ax_shap.invert_yaxis()
        ax_shap.set_xlabel("Net SHAP Value")
        ax_freq.set_xlabel("Word Frequency (Scaled)")

        max_limit = max(np.max(np.abs(shap_vals)), np.max(np.abs(mirrored_freqs))) * 1.2
        ax_shap.set_xlim(-max_limit, max_limit)
        ax_freq.set_xlim(-max_limit, max_limit)

        ax_shap.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax_freq.axvline(0, color='gray', linestyle='--', linewidth=1)

        ax_shap.legend(loc='upper left')
        plt.title("Top 20 Words (Excl. Stopwords): SHAP Impact Overlaid on True Frequency")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error creating SHAP-frequency plot: {e}")




class FittedKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, full_X):
        self.model = model
        self.full_X = full_X  # Must match order of tabular X
        self.is_fitted_ = False

    def fit(self, X, y):
        self._n_features_in_ = X.shape[1]
        self.y_train = y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        idxs = np.arange(X.shape[0])
        return self.model.predict(self.full_X[idxs]).flatten()

with st.expander("Partial Dependence Plot (PDP) for Tabular Features"):
    try:
        # Step 1: Define tabular features
        tabular_features = [
            'host_is_superhost', 'latitude', 'longitude',
            'accommodates', 'bathrooms', 'bedrooms', 'beds',
            'minimum_nights', 'number_of_reviews', 'review_scores_rating',
            'calculated_host_listings_count'
        ]

        # Step 2: Extract data
        X_tabular = X_train_dl.toarray()[:, :len(tabular_features)]
        y_tabular = y_train_dl[:X_tabular.shape[0]]

        # Step 3: Train a model on tabular features
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_tabular, y_tabular)

        # --- 1D PDP ---
        st.subheader("📈 Select a Feature for 1D PDP")
        selected_feature = st.selectbox("Choose a feature", tabular_features)

        fig1d, ax1d = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(
            estimator=rf_model,
            X=X_tabular,
            features=[tabular_features.index(selected_feature)],
            feature_names=tabular_features,
            ax=ax1d,
            kind="average",
            grid_resolution=50
        )
        st.pyplot(fig1d)

        # --- 2D PDP ---
        st.subheader("🧩 Select Feature Pair for 2D PDP")
        feature1 = st.selectbox("Feature 1", tabular_features, index=4)
        feature2 = st.selectbox("Feature 2", tabular_features, index=5)

        if feature1 != feature2:
            fig2d, ax2d = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(
                estimator=rf_model,
                X=X_tabular,
                features=[(tabular_features.index(feature1), tabular_features.index(feature2))],
                feature_names=tabular_features,
                ax=ax2d,
                kind="average",
                grid_resolution=30
            )
            st.pyplot(fig2d)
        else:
            st.warning("Please select two different features for 2D interaction plot.")

    except Exception as e:
        st.error(f"Error generating PDP: {e}")