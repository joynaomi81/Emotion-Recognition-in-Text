import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the pre-trained model
pipe_lr = joblib.load(open("model.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😡",  
    "disgust": "🤢",  
    "fear": "😱",  
    "happy": "😊",  
    "joy": "😄",  
    "neutral": "😶",  
    "sad": "😢",  
    "sadness": "😔",  
    "shame": "😳",  
    "surprise": "😮",  
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detector")
    st.subheader("Detect Emotions in Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "❓")  
            st.write("{}: {}".format(prediction.capitalize(), emoji_icon))  
            st.write("Confidence: {:.2f}".format(np.max(probability)))  

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotions", "Probability"]

            # Create a bar chart with Altair
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('Emotions', sort='-y'),  # Sort by probability
                y='Probability',
                color='Emotions'
            ).properties(title="Emotion Prediction Probability")

            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
