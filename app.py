import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import openai

# Embed dataset
csv_data = """Disease,Description,Symptoms,Precautions,Medicines,Dosage,Emergency_Signs,Lifestyle_Tips
Common Cold,A viral infection of your nose and throat.,cough,sneeze,congestion,rest,hydration,paracetamol,1 tab thrice a day,severe chest pain,Stay hydrated, rest well
Flu,An infectious disease caused by the influenza virus.,fever,cough,chills,tamiflu,oseltamivir,1 tab twice a day,difficulty breathing,Avoid cold drinks, wear warm clothes
Malaria,A disease caused by a plasmodium parasite.,fever,shivering,sweating,chloroquine,1 tab twice a day,delirium or unconsciousness,Use mosquito nets, clean water
Typhoid,A bacterial infection due to Salmonella.,fever,abdominal pain,weakness,azithromycin,1 tab once daily,severe dehydration,Boil drinking water, avoid raw food
COVID-19,Respiratory infection by coronavirus.,fever,cough,loss of taste,paracetamol,zinc,1 tab thrice a day,low oxygen,Isolate, wear mask, take vitamins
Asthma,Airways inflamed and narrowed.,cough,short breath,wheezing,albuterol inhaler,2 puffs as needed,cyanosis,Avoid allergens, use inhaler regularly
Diabetes,High blood sugar levels.,fatigue,thirst,urination,metformin,500mg twice daily,confusion or fainting,Monitor sugar, avoid sweets
Hypertension,High blood pressure.,headache,dizziness,nosebleed,amlodipine,5mg once daily,blurred vision,Exercise, low-salt diet
Migraine,Severe recurrent headache.,headache,nausea,light sensitivity,sumatriptan,50mg as needed,vision loss,Avoid triggers, rest in dark
"""

# Convert CSV to DataFrame
from io import StringIO
df = pd.read_csv(StringIO(csv_data))

# Preprocess
df['Symptoms'] = df['Symptoms'].str.split(',')
X = df['Symptoms'].apply(lambda x: pd.Series(x)).fillna('')
X.columns = [f'Symptom_{i+1}' for i in range(X.shape[1])]
X = X.apply(lambda col: col.astype(str).str.lower())

y = df['Disease']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X.apply(lambda x: ''.join(x), axis=1).apply(hash).values.reshape(-1, 1), y_encoded)

# UI
st.set_page_config(page_title="AI Health Advisor", layout="centered")
st.title("🤖 AI Medicine Recommendation System")

symptoms = st.text_input("Enter your symptoms (comma-separated):").lower().split(',')

if st.button("Predict Disease"):
    input_data = ''.join([s.strip() for s in symptoms])
    input_hash = hash(input_data)
    prediction = model.predict([[input_hash]])
    predicted_disease = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Disease: {predicted_disease}")

    info = df[df['Disease'] == predicted_disease].iloc[0]
    tabs = st.tabs(["🩺 Description", "💊 Medicine", "💡 Precautions", "🕒 Dosage", "🚨 Emergency Signs", "🌿 Lifestyle Tips"])

    tabs[0].write(info['Description'])
    tabs[1].write(info['Medicines'])
    tabs[2].write(info['Precautions'])
    tabs[3].write(info['Dosage'])
    tabs[4].write(info['Emergency_Signs'])
    tabs[5].write(info['Lifestyle_Tips'])

# Chatbot
st.markdown("---")
st.subheader("💬 Ask our AI Health Bot")

openai.api_key = st.secrets["openai_key"] if "openai_key" in st.secrets else "sk-..."

user_q = st.text_input("Ask a health-related question:")
if st.button("Get Answer"):
    if "sk-" not in openai.api_key:
        st.warning("⚠️ Please add your OpenAI API key in Streamlit secrets to use the chatbot.")
    else:
        with st.spinner("Thinking..."):
            try:
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_q}],
                    temperature=0.5,
                    max_tokens=150
                )
                st.info(res.choices[0].message['content'])
            except Exception as e:
                st.error(f"Error: {e}")
