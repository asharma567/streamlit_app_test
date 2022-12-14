#create the explainer and shap values 

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import shap
import streamlit as st
import streamlit.components.v1 as components
import pickle
import xgboost
import joblib




def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def scale_and_standardize(df):
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    return scaled_df

def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

st.title("Boba Leaderboard")

#load joblib
df_M_character = pickle.load(open('df_M_character.pickle', 'rb'))
# explainer = pickle.load(open('explainer.pickle', 'rb'))
# shap_values = pickle.load(open('shap_values.pickle', 'rb'))
df_M_character_scale = scale_and_standardize(df_M_character)
author_idx_lookup = dict([(name, idx) for idx, name in enumerate(df_M_character.index)])
ppl_labels = joblib.load('ppl_labels.joblib') 
# train XGBoost model
X, y = pd.DataFrame(
    df_M_character_scale, 
    columns=df_M_character.columns
), ppl_labels

bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X)
# shap.summary_plot(shap_values, X)


X = pd.DataFrame(
    df_M_character_scale, 
    columns=df_M_character.columns)


df_leaderboard = pd.read_csv("leaderboard.csv")
df_leaderboard.rename(columns = {'Unnamed: 0':'Rank', 'author':'member'}, inplace=True)
selection = aggrid_interactive_table(df=df_leaderboard)


if selection:
    try: 
        auth = selection["selected_rows"][0]['member']
    except:
        auth = df_leaderboard['member'].iloc[0]

    st.write(auth)
    author_idx = author_idx_lookup.get(auth)
    # visualize the first prediction's explaination with default colors
    st_shap(shap.force_plot(explainer.expected_value, shap_values[author_idx,:], X.iloc[author_idx,:]))







