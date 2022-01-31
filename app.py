import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

# --------------------------------- #
# Page layout
# Expand to full width
st.set_page_config(
    page_title="Machine Learning App",
    layout="wide"
)

# --------------------------------- #
# Model
def build_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:,-1]

    # Data spliting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_size)/ 100)

    st.markdown("**1.2 Data samples**")
    st.write("Shape of Training sample:")
    st.info(X.shape)
    st.write("Shape of Test sample:")
    st.info(y.shape)

    st.markdown("**1.3. Variable details**:")
    st.write("X variable")
    st.info(list(X.columns))
    st.write("Y variable")
    st.info(y.name)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_features=max_features,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs
    )

    rf.fit(X_train, y_train)

    st.subheader("2. Model Performance:")

    st.markdown("**2.1. Training set:**")
    y_pred_train = rf.predict(X_train)

    st.write("Coefficient of determination ($R^2$):")
    st.info(r2_score(y_train, y_pred_train))

    st.write("Error (MSE or MAE):")
    st.info(mean_squared_error(y_train, y_pred_train))

    st.markdown("**2.2. Test set:**")
    y_pred_test = rf.predict(X_train)

    st.write("Coefficient of determination ($R^2$):")
    st.info(r2_score(y_train, y_pred_test))

    st.write("Error (MSE or MAE):")
    st.info(mean_squared_error(y_train, y_pred_test))

    st.subheader('3. Model Parameters:')
    st.write(rf.get_params())

# --------------------------------- #
# Sidebar
with st.sidebar.header("1. Upload your CSV file:"):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file",
        type=[
            "csv"
        ]
    )
    st.sidebar.markdown("""
        [Example CSV file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

with st.sidebar.header("2. Set Parameters:"):
    split_size = st.sidebar.slider("Data split ratio (% for Training Set):", 10, 90, 80, 5)

with st.sidebar.header("2.1 Learning parameters:"):
    n_estimators = st.sidebar.slider("Number of estimators (n_estimators)", 0, 1000, 100, 100)
    max_features = st.sidebar.select_slider("Max features (max_features)", options=["auto", "sqrt", "log2"])
    min_samples_split = st.sidebar.slider("Minimum number of samples required to split an internal node (min_samples_split):", 1, 10, 2, 1)
    min_samples_leaf = st.sidebar.slider("Minimum number of samples required to be at a lead node (min_samples_leaf):", 1, 10, 2, 1)

with st.sidebar.subheader("2.2 General parameters:"):
    random_state = st.sidebar.slider("Random seed: ", 0, 1000, 42, 1)
    criterion = st.sidebar.select_slider("Performance measure (criterion):", options=["mse", "mae"])
    bootstrap = st.sidebar.select_slider("Bootstrap samples when building trees (bootstrap):", options=[True, False])
    oob_score = st.sidebar.select_slider("Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score):", options=[False, True])
    n_jobs = st.sidebar.select_slider("Number of jobs to run in parallel (n_jobs):", options=[-1, 1])


# --------------------------------- #
# Main panel
# Display data
st.subheader("1. Dataset")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("**1.1 A glimpse of data**")
    st.write(df)
    build_model()
else:
    st.info("CSV file to be uploaded")
    if st.button("Run to use examplary Data"):
        # Boston housing data
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target, name="response")
        df = pd.concat([X, y], axis=1)

        st.markdown("The Boston housing data is used as examplay data:")
        st.write(df.head(5))

        build_model(df)
