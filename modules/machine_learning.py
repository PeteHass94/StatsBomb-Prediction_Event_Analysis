import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import streamlit as st
import modules.visuals as visuals
import time

# Feature types
binned_columns = [
    ("binned_xg_for", "binned_xg_against"),
    ("binned_final_third_passes_for", "binned_final_third_passes_against"),
    ("binned_box_passes_for", "binned_box_passes_against"),
    ("binned_final_third_carries_for", "binned_final_third_carries_against"),
    ("binned_box_carries_for", "binned_box_carries_against")
]

rolling_columns = [
    ("rolling_xg_for", "rolling_xg_against"),
    ("rolling_final_third_passes_for", "rolling_final_third_passes_against"),
    ("rolling_box_passes_for", "rolling_box_passes_against"),
    ("rolling_final_third_carries_for", "rolling_final_third_carries_against"),
    ("rolling_box_carries_for", "rolling_box_carries_against")
]

# Flatten column names
binned_columns_flat = [col for pair in binned_columns for col in pair]
rolling_columns_flat = [col for pair in rolling_columns for col in pair]
feature_columns = binned_columns_flat + rolling_columns_flat


def explode_rows_per_bin(match_row, bin_width=10):
    """
    Turns each match row into 9 rows, one per bin (0 to 8),
    using cumulative features up to that bin and label from bin+1 to bin+10.
    """
    rows = []
    for bin_idx in range(9):  # bins 0 to 8
        row = {
            'match_id': match_row['match_id'],
            'team': match_row['team'],
            'team_match_summary': match_row['team_match_summary'],
            'bin': bin_idx,
            'bin_range': f"{(bin_idx + 1) * bin_width}-{(bin_idx + 2) * bin_width}"
        }

        for col in feature_columns:
            values = match_row[col]
            for i in range(bin_idx + 1):  # use data up to and including current bin
                row[f"{col}_{i}"] = values[i]

        # Add target labels
        row["big_chance_for_next_10"] = match_row["big_chance_for_next_10"][bin_idx]
        row["big_chance_against_next_10"] = match_row["big_chance_against_next_10"][bin_idx]
        rows.append(row)

    return rows


def prepare_dataset(matches_df):
    """
    Expand matches_df into 9x more rows, one per bin per match (bins 0–8).
    """
    all_rows = []
    for _, match_row in matches_df.iterrows():
        all_rows.extend(explode_rows_per_bin(match_row))
    return pd.DataFrame(all_rows)


def get_train_test_split_indices(df, match_id_col='match_id', test_ratio=0.2):
    """
    Perform 80/20 split on full matches.
    """
    match_ids = df[match_id_col].unique()
    np.random.seed(42)
    np.random.shuffle(match_ids)
    split_idx = int(len(match_ids) * (1 - test_ratio))
    train_matches = set(match_ids[:split_idx])
    test_matches = set(match_ids[split_idx:])

    train_idx = df[df[match_id_col].isin(train_matches)].index
    test_idx = df[df[match_id_col].isin(test_matches)].index
    
    # Optional visualization for debugging 
    safe_columns = [col for col in [
        "match_id", "bin_range", "team_match_summary", "big_chance_for_next_10", "big_chance_against_next_10"
    ] if col in df.columns]

    st.subheader("📊 Match Bin Split")
    st.markdown(f"### Train Match Bins ({len(train_idx)} rows)")
    st.dataframe(df.loc[train_idx, safe_columns])
    st.markdown(f"### Test Match Bins ({len(test_idx)} rows)")
    st.dataframe(df.loc[test_idx, safe_columns])

    return train_idx, test_idx


def train_and_evaluate_model(df, target_col):
    """
    Train and evaluate model using the expanded bin-level dataset.
    """
    target_cols = [f"{target_col}_for_next_10", f"{target_col}_against_next_10"]
    labelsTitle = target_col.replace("_", " ").title()
    labels = (labelsTitle + " For", labelsTitle + " Against") 
    
    #get features  
    features = [col for col in df.columns if col.startswith("binned_") or col.startswith("rolling_")]

    train_idx, test_idx = get_train_test_split_indices(df)
    
    st.subheader("📈 Model Training and Evaluation")
    
    with st.spinner("Processing...", show_time=True):
        time.sleep(60)
    
    for t_col, lab in zip(target_cols, labels):
    
        X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, t_col]
        X_test, y_test = df.loc[test_idx, features], df.loc[test_idx, t_col]

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        st.text(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")
        st.text(f"Test Accuracy: {model.score(X_test, y_test):.2f}")
        st.text(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
        st.text("Classification Report:")
        st.code(classification_report(y_test, model.predict(X_test)))

        # Prediction sample
        test_df = df.loc[test_idx].copy()
        test_df["Predicted_Probability"] = model.predict_proba(X_test)[:, 1]
        test_df["Predicted_Label"] = model.predict(X_test)
        test_df["True_Label"] = y_test.values
        test_df["Accuracy"] = (test_df["Predicted_Label"] == test_df["True_Label"]).astype(int)
        
        # Group by match
        grouped = (
            test_df
            .groupby(["match_id", "team", "team_match_summary"])
            .agg(
                avg_predicted_prob=("Predicted_Probability", "mean"),
                bin_accuracy=("Predicted_Label", lambda x: (x == test_df.loc[x.index, "True_Label"]).mean()),
                total_bins=("Predicted_Label", "count"),
                predicted_big_chances=("Predicted_Label", "sum"),  # count of predicted 1s
                actual_big_chances=("True_Label", "sum")           # count of actual 1s
            )
            .reset_index()
            .sort_values("avg_predicted_prob", ascending=False)
        )
        
        st.subheader(f"📈 Per-Match Summary: {lab}")
        st.markdown(f"### All Test Matches ({len(grouped)} rows)")
        st.dataframe(grouped)
        
        all_teams = test_df["team"].unique()
        selected_team_name1 = st.selectbox("🎯 Select a team to see their match predictions:", sorted(all_teams),
                                           key=f"team_select_{t_col}")  # sort for easier UX
        grouped_team = grouped[grouped["team"] == selected_team_name1]
        st.markdown(f"### {selected_team_name1} Matches ({len(grouped_team)} rows)")
        st.dataframe(grouped_team)
        # st.text(test_df.columns)      
        
        # Detailed match viewer
        match_sample = st.selectbox(f"🎯 Select a test match for `{lab}`:", grouped_team['team_match_summary'].unique(),
                                    key=f"match_select_{t_col}")
        sample = test_df[test_df['team_match_summary'] == match_sample][[
            'match_id', 'team_match_summary', 'bin_range', 'Predicted_Probability', 'Predicted_Label', 'True_Label', 'Accuracy'
        ]]
        st.dataframe(sample)
        visuals.plot_gantt_chart(sample, lab)
    # return model


def train_big_chance_prediction_models(matches_df):
    """
    Pipeline entry point: trains two models using bin-level event data.
    """
    st.header("🧠 Big Chance Prediction")

    st.info("🚀 Expanding match rows into per-bin training data...")
    exploded_df = prepare_dataset(matches_df)

    # model_for = train_and_evaluate_model(exploded_df, "big_chance_for_next_10", label="Big Chance For")
    # model_against = train_and_evaluate_model(exploded_df, "big_chance_against_next_10", label="Big Chance Against")

    train_and_evaluate_model(exploded_df, "big_chance")
    
    # return model_for, model_against

