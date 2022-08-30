import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost
import joblib




def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.title("Boba Leaderboard")

def scale_and_standardize(df):
    from sklearn.preprocessing import StandardScaler    
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    return scaled_df

#load joblib
df_M_character = joblib.load('../df_M_character.joblib')
df_M_character_scale = scale_and_standardize(df_M_character)
author_idx_lookup = dict([(name, idx) for idx, name in enumerate(df_M_character.index)])

explainer = joblib.load('../explainer.joblib')
shap_values = joblib.load('../shap_values.joblib')


X = pd.DataFrame(
    df_M_character_scale, 
    columns=df_M_character.columns)



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

#change to leaderboard
leaderboard = pd.read_csv("leaderboard.csv")

selection = aggrid_interactive_table(df=leaderboard)

if selection:


    auth = selection["selected_rows"][0]['author']
    st.write(auth)
    author_idx = author_idx_lookup.get(auth)


    # visualize the first prediction's explaination with default colors
    st_shap(shap.force_plot(explainer.expected_value, shap_values[author_idx,:], X.iloc[author_idx,:]))



