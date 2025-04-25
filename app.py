import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import random
from io import StringIO # Import StringIO from io


# ------------------------
# Dataset
# ------------------------
"Disease","Description","Symptoms","Precautions","Medicines","Dosage","Emergency_Signs","Lifestyle_Tips"
"Common Cold","A viral infection of your nose and throat.","cough,sneeze,congestion","rest,hydration","paracetamol","1 tab thrice a day","severe chest pain","Stay hydrated, rest well"
"Flu","An infectious disease caused by the influenza virus.","fever,cough,chills","rest,fluids","tamiflu,oseltamivir","1 tab twice a day","difficulty breathing","Avoid cold drinks, wear warm clothes"
"Malaria","A disease caused by a plasmodium parasite.","fever,shivering,sweating","mosquito nets,clean water","chloroquine","1 tab twice a day","delirium or unconsciousness","Use mosquito nets, clean water"
"Typhoid","A bacterial infection due to Salmonella.","fever,abdominal pain,weakness","boil water,avoid raw food","azithromycin","1 tab once daily","severe dehydration","Boil drinking water, avoid raw food"
"Dengue","A mosquito-borne viral disease.","fever,joint pain,rash","rest,fluids","paracetamol","1 tab thrice a day","internal bleeding","Avoid NSAIDs, rest & fluids"
"COVID-19","Respiratory infection by coronavirus.","fever,cough,loss of taste","isolate,mask","paracetamol,zinc","1 tab thrice a day","low oxygen","Isolate, wear mask, take vitamins"
"Asthma","Airways inflamed and narrowed.","cough,short breath,wheezing","avoid allergens","albuterol inhaler","2 puffs as needed","cyanosis","Avoid allergens, use inhaler regularly"
"Diabetes","High blood sugar levels.","fatigue,thirst,urination","no sugar,check levels","metformin","500mg twice daily","confusion or fainting","Monitor sugar, avoid sweets"
"Hypertension","High blood pressure.","headache,dizziness,nosebleed","low salt diet","amlodipine","5mg once daily","blurred vision","Exercise, low-salt diet"
"Migraine","Severe recurrent headache.","headache,nausea,light sensitivity","avoid triggers","sumatriptan","50mg as needed","vision loss","Avoid triggers, rest in dark"
"Pneumonia","Infection inflaming lungs.","cough,fever,short breath","rest,antibiotics","azithromycin","500mg once daily","blue lips or nails","Breathing exercises, fluids"
"TB","Infectious lung disease.","cough,fever,weight loss","mask,medication","isoniazid","300mg daily","hemoptysis","Regular treatment, wear mask"
"Allergy","Immune reaction to substances.","sneezing,rash,itching","avoid allergens","antihistamines","10mg once daily","anaphylaxis","Avoid allergens, use antihistamines"
"Bronchitis","Inflamed bronchial tubes.","cough,mucus,wheezing","no smoke,hydration","amoxicillin","500mg thrice daily","chest tightness","Avoid smoke, stay hydrated"
"Sinusitis","Sinus inflammation.","facial pain,nasal block,congestion","steam,rest","antibiotics","1 tab daily","fever with swelling","Use saline spray, rest"
"Conjunctivitis","Inflamed eye conjunctiva.","red eye,itchy eyes,tears","no eye rubbing","chloramphenicol drops","1 drop thrice daily","vision loss","Avoid touching eyes"
"UTI","Bacterial infection of urinary tract.","burning urine,fever,urgency","hydration,clean habits","trimethoprim","100mg twice daily","flank pain","Hydrate well, avoid irritants"
"Anemia","Low red blood cell count.","fatigue,pale skin,dizziness","iron rich food","iron supplements","1 tab daily","chest pain","Iron-rich foods, rest"
"Gastritis","Stomach lining inflammation.","nausea,bloating,stomach pain","light meals","omeprazole","20mg once daily","black stools","Light food, avoid spicy items"
"Acidity","Excess acid in stomach.","burning chest,sour belch,indigestion","avoid late meals","antacids","1 tab after food","severe chest pain","Avoid late meals, eat small portions"
"Jaundice","Yellowing of skin/liver issue.","yellow eyes,fatigue,nausea","avoid fat food","liver tonics","as per dose","extreme weakness","Avoid oily food, hydrate well"
"Chickenpox","Viral infection with rash.","rash,fever,itching","rest,avoid scratching","calamine,antivirals","as per dose","high fever","Avoid scratching, rest"
"Measles","Highly contagious virus.","fever,rash,runny nose","isolate,hygiene","vitamin A","once daily","breathing issues","Isolate, good hygiene"
"Mumps","Viral swelling of glands.","jaw pain,fever,swollen cheeks","ice packs","painkillers","as needed","difficulty swallowing","Ice pack, avoid sour food"
"Scabies","Skin infestation by mites.","itchy rash,burrows","clean clothes","permethrin cream","apply overnight","secondary infection","Wash clothes, cut nails"
"Eczema","Inflamed itchy skin.","red rash,itching,dryness","avoid triggers","moisturizer,steroids","as per dose","severe inflammation","Avoid allergens"
"Psoriasis","Autoimmune skin disorder.","scaly skin,red patches,itching","moisturize,UV therapy","steroid creams","apply twice daily","joint pain","Use lotion, light therapy"
"Acne","Blocked skin pores.","pimples,oily skin,scars","wash face,avoid oil","benzoyl peroxide","apply daily","severe cysts","Avoid oily food, wash face"
"Depression","Mood disorder.","sadness,low energy,sleep changes","therapy,healthy food","SSRIs","as per doctor","suicidal thoughts","Talk therapy, regular schedule"
"Anxiety","Feeling of fear/nervousness.","palpitations,sweating,worry","meditation,no caffeine","anxiolytics","as per doctor","chest tightness","Meditation, reduce caffeine"
"Arthritis","Joint inflammation.","joint pain,stiffness,swelling","mild exercise","NSAIDs","as needed","limited motion","Hot pads, mild exercise"
"Back Pain","Muscle or spine issues.","lower pain,stiffness,numbness","stretch,posture","painkillers","as needed","leg weakness","Stretch, posture correction"
"Obesity","Excess body fat.","weight gain,fatigue,breathlessness","exercise,diet","orlistat","120mg with meals","shortness of breath","Exercise, diet control"
"PCOS","Hormonal disorder.","irregular periods,acne,weight gain","healthy diet","hormonal meds","as per doctor","extreme pain","Healthy diet, weight management"
"Menstrual Cramps","Pain during periods.","abdominal pain,nausea,fatigue","heat pad,relax","ibuprofen","400mg as needed","severe bleeding","Hot water bag, relax"
"Appendicitis","Inflamed appendix.","abdominal pain,fever,nausea","surgery required","NA","NA","rupture or shock","Immediate medical help"
"Tonsillitis","Swollen tonsils.","sore throat,fever,difficulty swallowing","lozenges,warm fluids","penicillin","500mg daily","trouble breathing","Use lozenges, warm fluids"
"Vertigo","Spinning sensation.","dizziness,nausea,balance issues","get up slowly","betahistine","16mg daily","fall risk","Change positions slowly"
"Ear Infection","Middle ear inflammation.","earache,fever,hearing loss","no objects in ear","amoxicillin","500mg daily","ear discharge","Avoid inserting objects"
"Insomnia","Difficulty sleeping.","sleeplessness,anxiety,tiredness","no screens before bed","melatonin","3mg at night","extreme fatigue","No caffeine, bedtime routine"
"GERD","Acid reflux.","heartburn,bloating,belching","no spicy food","ranitidine","150mg daily","chest pain","Elevate head, avoid late meals"
"Lactose Intolerance","Can't digest dairy.","bloating,diarrhea,gas","avoid dairy","lactase tablets","as per use","extreme cramps","Avoid dairy, use alternatives"

# Use StringIO to load the CSV data
df = pd.read_csv(StringIO(csv_data)) # Changed to StringIO from io
# Convert the 'Symptoms' column to string type before applying the split
df["Symptoms"] = df["Symptoms"].astype(str).apply(lambda x: x.split(","))

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
y = df["Disease"]

model = RandomForestClassifier()
model.fit(X, y)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config("AI Health Advisor", layout="centered")
st.title("🧠 AI Health Advisor")
st.markdown("Your smart assistant for predicting disease and health guidance.")

# ------------------------
# Health Tip Carousel
# ------------------------
health_tips = [
    "💧 Stay hydrated – drink 8+ glasses of water daily.",
    "🥦 Eat more fiber-rich fruits and vegetables.",
    "😴 Aim for 7–8 hours of quality sleep every night.",
    "🚶 Take a brisk 30-min walk daily for heart health.",
    "🧼 Wash hands frequently to avoid infections.",
    "🧘 Practice deep breathing or meditation for 10 mins.",
    "💊 Don’t self-medicate without expert advice.",
    "🧴 Use mosquito repellents to prevent vector diseases.",
    "🧻 Keep surroundings clean to avoid waterborne diseases.",
]
st.info(f"💡 Health Tip: **{random.choice(health_tips)}**")

# ------------------------
# Symptom Input
# ------------------------
all_symptoms = sorted(set(sym.strip() for sublist in df["Symptoms"] for sym in sublist))
selected = st.multiselect("🤒 Select your symptoms", all_symptoms)

if st.button("🔍 Predict Disease"):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        input_vec = mlb.transform([selected])
        result = model.predict(input_vec)[0]
        data = df[df["Disease"] == result].iloc[0]

        st.success(f"🩺 Possible Disease: **{result}**")
        st.markdown(f"**About:** {data['Description']}")

        tab1, tab2, tab3, tab4 = st.tabs(["💊 Medicine", "🥗 Precautions", "⚠️ Emergency", "🌿 Lifestyle"])

        with tab1:
            st.write(f"**Medicine:** {data['Medicines']}")
            st.write(f"**Dosage:** {data['Dosage']}")

        with tab2:
            st.write(f"**Precautions:** {data['Precautions']}")

        with tab3:
            st.write(f"**Emergency Signs:** {data['Emergency_Signs']}")

        with tab4:
            st.write(f"**Lifestyle Tips:** {data['Lifestyle_Tips']}")

# ------------------------
# Chatbot
# ------------------------
import openai

st.header("🤖 GPT-powered Health Chat")

api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

user_query = st.text_area("💬 Ask me anything about your symptoms, disease, or medicine:")

if st.button("Ask GPT"):
    if not api_key or not user_query:
        st.warning("Please provide both the API key and a question.")
    else:
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant. Respond clearly and simply."},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.7,
                max_tokens=200
            )
            st.success(response.choices[0].message["content"])
        except Exception as e:
            st.error(f"Error: {str(e)}")

