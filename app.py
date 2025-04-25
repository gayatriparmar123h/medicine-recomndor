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
dataset = """
Disease: Common Cold
Description: A viral infection of your nose and throat.
Symptoms: cough, sneeze, congestion
Precautions: rest, hydration
Medicines: paracetamol
Dosage: 1 tab thrice a day
Emergency Signs: severe chest pain
Lifestyle Tips: Stay hydrated, rest well

Disease: Flu
Description: An infectious disease caused by the influenza virus.
Symptoms: fever, cough, chills
Precautions: rest, fluids
Medicines: tamiflu, oseltamivir
Dosage: 1 tab twice a day
Emergency Signs: difficulty breathing
Lifestyle Tips: Avoid cold drinks, wear warm clothes

Disease: Malaria
Description: A disease caused by a plasmodium parasite.
Symptoms: fever, shivering, sweating
Precautions: mosquito nets, clean water
Medicines: chloroquine
Dosage: 1 tab twice a day
Emergency Signs: delirium or unconsciousness
Lifestyle Tips: Use mosquito nets, clean water

Disease: Typhoid
Description: A bacterial infection caused by Salmonella.
Symptoms: fever, abdominal pain, weakness
Precautions: boil water, avoid raw food
Medicines: azithromycin
Dosage: 1 tab once daily
Emergency Signs: severe dehydration
Lifestyle Tips: Boil drinking water, avoid raw food

Disease: Dengue
Description: A mosquito-borne viral disease.
Symptoms: fever, joint pain, rash
Precautions: rest, fluids
Medicines: paracetamol
Dosage: 1 tab thrice a day
Emergency Signs: internal bleeding
Lifestyle Tips: Avoid NSAIDs, rest & fluids

Disease: COVID-19
Description: A respiratory infection caused by the coronavirus.
Symptoms: fever, cough, loss of taste
Precautions: isolate, wear mask
Medicines: paracetamol, zinc
Dosage: 1 tab thrice a day
Emergency Signs: low oxygen
Lifestyle Tips: Isolate, wear mask, take vitamins

Disease: Asthma
Description: Airways inflamed and narrowed.
Symptoms: cough, short breath, wheezing
Precautions: avoid allergens
Medicines: albuterol inhaler
Dosage: 2 puffs as needed
Emergency Signs: cyanosis
Lifestyle Tips: Avoid allergens, use inhaler regularly

Disease: Diabetes
Description: High blood sugar levels.
Symptoms: fatigue, thirst, urination
Precautions: no sugar, check levels
Medicines: metformin
Dosage: 500mg twice daily
Emergency Signs: confusion or fainting
Lifestyle Tips: Monitor sugar, avoid sweets

Disease: Hypertension
Description: High blood pressure.
Symptoms: headache, dizziness, nosebleed
Precautions: low salt diet
Medicines: amlodipine
Dosage: 5mg once daily
Emergency Signs: blurred vision
Lifestyle Tips: Exercise, low-salt diet

Disease: Migraine
Description: Severe recurrent headache.
Symptoms: headache, nausea, light sensitivity
Precautions: avoid triggers
Medicines: sumatriptan
Dosage: 50mg as needed
Emergency Signs: vision loss
Lifestyle Tips: Avoid triggers, rest in dark

Disease: Pneumonia
Description: Infection inflaming the lungs.
Symptoms: cough, fever, short breath
Precautions: rest, antibiotics
Medicines: azithromycin
Dosage: 500mg once daily
Emergency Signs: blue lips or nails
Lifestyle Tips: Breathing exercises, fluids

Disease: TB
Description: Infectious lung disease.
Symptoms: cough, fever, weight loss
Precautions: mask, medication
Medicines: isoniazid
Dosage: 300mg daily
Emergency Signs: hemoptysis
Lifestyle Tips: Regular treatment, wear mask

Disease: Allergy
Description: Immune reaction to substances.
Symptoms: sneezing, rash, itching
Precautions: avoid allergens
Medicines: antihistamines
Dosage: 10mg once daily
Emergency Signs: anaphylaxis
Lifestyle Tips: Avoid allergens, use antihistamines

Disease: Bronchitis
Description: Inflammation of bronchial tubes.
Symptoms: cough, mucus, wheezing
Precautions: no smoke, hydration
Medicines: amoxicillin
Dosage: 500mg thrice daily
Emergency Signs: chest tightness
Lifestyle Tips: Avoid smoke, stay hydrated

Disease: Sinusitis
Description: Inflammation of the sinuses.
Symptoms: facial pain, nasal block, congestion
Precautions: steam, rest
Medicines: antibiotics
Dosage: 1 tab daily
Emergency Signs: fever with swelling
Lifestyle Tips: Use saline spray, rest

Disease: Conjunctivitis
Description: Inflammation of eye conjunctiva.
Symptoms: red eye, itchy eyes, tears
Precautions: no eye rubbing
Medicines: chloramphenicol drops
Dosage: 1 drop thrice daily
Emergency Signs: vision loss
Lifestyle Tips: Avoid touching eyes

Disease: UTI
Description: Bacterial infection of urinary tract.
Symptoms: burning urine, fever, urgency
Precautions: hydration, clean habits
Medicines: trimethoprim
Dosage: 100mg twice daily
Emergency Signs: flank pain
Lifestyle Tips: Hydrate well, avoid irritants

Disease: Anemia
Description: Low red blood cell count.
Symptoms: fatigue, pale skin, dizziness
Precautions: iron-rich food
Medicines: iron supplements
Dosage: 1 tab daily
Emergency Signs: chest pain
Lifestyle Tips: Iron-rich foods, rest

Disease: Gastritis
Description: Stomach lining inflammation.
Symptoms: nausea, bloating, stomach pain
Precautions: light meals
Medicines: omeprazole
Dosage: 20mg once daily
Emergency Signs: black stools
Lifestyle Tips: Light food, avoid spicy items

Disease: Acidity
Description: Excess acid in the stomach.
Symptoms: burning chest, sour belch, indigestion
Precautions: avoid late meals
Medicines: antacids
Dosage: 1 tab after food
Emergency Signs: severe chest pain
Lifestyle Tips: Avoid late meals, eat small portions

Disease: Jaundice
Description: Yellowing of skin/liver issue.
Symptoms: yellow eyes, fatigue, nausea
Precautions: avoid fatty food
Medicines: liver tonics
Dosage: as per dose
Emergency Signs: extreme weakness
Lifestyle Tips: Avoid oily food, hydrate well

Disease: Chickenpox
Description: Viral infection with rash.
Symptoms: rash, fever, itching
Precautions: rest, avoid scratching
Medicines: calamine, antivirals
Dosage: as per dose
Emergency Signs: high fever
Lifestyle Tips: Avoid scratching, rest

Disease: Measles
Description: Highly contagious virus.
Symptoms: fever, rash, runny nose
Precautions: isolate, hygiene
Medicines: vitamin A
Dosage: once daily
Emergency Signs: breathing issues
Lifestyle Tips: Isolate, good hygiene

Disease: Mumps
Description: Viral swelling of glands.
Symptoms: jaw pain, fever, swollen cheeks
Precautions: ice packs
Medicines: painkillers
Dosage: as needed
Emergency Signs: difficulty swallowing
Lifestyle Tips: Ice pack, avoid sour food

Disease: Scabies
Description: Skin infestation by mites.
Symptoms: itchy rash, burrows
Precautions: clean clothes
Medicines: permethrin cream
Dosage: apply overnight
Emergency Signs: secondary infection
Lifestyle Tips: Wash clothes, cut nails

Disease: Eczema
Description: Inflamed itchy skin.
Symptoms: red rash, itching, dryness
Precautions: avoid triggers
Medicines: moisturizer, steroids
Dosage: as per dose
Emergency Signs: severe inflammation
Lifestyle Tips: Avoid allergens

Disease: Psoriasis
Description: Autoimmune skin disorder.
Symptoms: scaly skin, red patches, itching
Precautions: moisturize, UV therapy
Medicines: steroid creams
Dosage: apply twice daily
Emergency Signs: joint pain
Lifestyle Tips: Use lotion, light therapy

Disease: Acne
Description: Blocked skin pores.
Symptoms: pimples, oily skin, scars
Precautions: wash face, avoid oil
Medicines: benzoyl peroxide
Dosage: apply daily
Emergency Signs: severe cysts
Lifestyle Tips: Avoid oily food, wash face

Disease: Depression
Description: Mood disorder.
Symptoms: sadness, low energy, sleep changes
Precautions: therapy, healthy food
Medicines: SSRIs
Dosage: as per doctor
Emergency Signs: suicidal thoughts
Lifestyle Tips: Talk therapy, regular schedule

Disease: Anxiety
Description: Feeling of fear/nervousness.
Symptoms: palpitations, sweating, worry
Precautions: meditation, no caffeine
Medicines: anxiolytics
Dosage: as per doctor
Emergency Signs: chest tightness
Lifestyle Tips: Meditation, reduce caffeine

Disease: Arthritis
Description: Joint inflammation.
Symptoms: joint pain, stiffness, swelling
Precautions: mild exercise
Medicines: NSAIDs
Dosage: as needed
Emergency Signs: limited motion
Lifestyle Tips: Hot pads, mild exercise

Disease: Back Pain
Description: Muscle or spine issues.
Symptoms: lower pain, stiffness, numbness
Precautions: stretch, posture
Medicines: painkillers
Dosage: as needed
Emergency Signs: leg weakness
Lifestyle Tips: Stretch, posture correction

Disease: Obesity
Description: Excess body fat.
Symptoms: weight gain, fatigue, breathlessness
Precautions: exercise, diet
Medicines: orlistat
Dosage: 120mg with meals
Emergency Signs: shortness of breath
Lifestyle Tips: Exercise, diet control

Disease: PCOS
Description: Hormonal disorder.
Symptoms: irregular periods, acne, weight gain
Precautions: healthy diet
Medicines: hormonal meds
Dosage: as per doctor
Emergency Signs: extreme pain
Lifestyle Tips: Healthy diet, weight management

Disease: Menstrual Cramps
Description: Pain during periods.
Symptoms: abdominal pain, nausea, fatigue
Precautions: heat pad, relax
Medicines: ibuprofen
Dosage: 400mg as needed
Emergency Signs: severe bleeding
Lifestyle Tips: Hot water bag, relax

Disease: Appendicitis
Description: Inflamed appendix.
Symptoms: abdominal pain, fever, nausea
Precautions: surgery required
Medicines: NA
Dosage: NA
Emergency Signs: rupture or shock
Lifestyle Tips: Immediate medical help

Disease: Tonsillitis
Description: Swollen tonsils.
Symptoms: sore throat, fever, difficulty swallowing
Precautions: lozenges, warm fluids
Medicines: penicillin
Dosage: 500mg daily
Emergency Signs: trouble breathing
Lifestyle Tips: Use lozenges, warm fluids

Disease: Vertigo
Description: Spinning sensation.
Symptoms: dizziness, nausea, balance issues
Precautions: get up slowly
Medicines: betahistine
Dosage: 16mg daily
Emergency Signs: fall risk
Lifestyle Tips: Change positions slowly

Disease: Ear Infection
Description: Middle ear inflammation.
Symptoms: earache, fever, hearing loss
Precautions: no objects in ear
Medicines: amoxicillin
Dosage: 500mg daily
Emergency Signs: ear discharge
Lifestyle Tips: Avoid inserting objects

Disease: Insomnia
Description: Difficulty sleeping.
Symptoms: sleeplessness, anxiety, tiredness
Precautions: no screens before bed
Medicines: melatonin
Dosage: 3mg at night
Emergency Signs: extreme fatigue
Lifestyle Tips: No caffeine, bedtime routine

Disease: GERD
Description: Acid reflux.
Symptoms: heartburn, bloating, belching
Precautions: no spicy food
Medicines: ranitidine
Dosage: 150mg daily
Emergency Signs: chest pain
Lifestyle Tips: Elevate head, avoid late meals

Disease: Lactose Intolerance
Description: Can't digest dairy.
Symptoms: bloating, diarrhea, gas
Precautions: avoid dairy
Medicines: lactase tablets
Dosage: as per use
Emergency Signs: extreme cramps
Lifestyle Tips: Avoid dairy, use alternatives
"""



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

api_key = st.text_input("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")

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

import streamlit as st
import openai
import pandas as pd

# Set up the OpenAI API key
openai.api_key = st.secrets["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"]  # Retrieve the API key from Streamlit secrets

# Dataset
dataset = """
Disease: Common Cold
Description: A viral infection of your nose and throat.
Symptoms: cough, sneeze, congestion
Precautions: rest, hydration
Medicines: paracetamol
Dosage: 1 tab thrice a day
Emergency Signs: severe chest pain
Lifestyle Tips: Stay hydrated, rest well

Disease: Flu
Description: An infectious disease caused by the influenza virus.
Symptoms: fever, cough, chills
Precautions: rest, fluids
Medicines: tamiflu, oseltamivir
Dosage: 1 tab twice a day
Emergency Signs: difficulty breathing
Lifestyle Tips: Avoid cold drinks, wear warm clothes

Disease: Malaria
Description: A disease caused by a plasmodium parasite.
Symptoms: fever, shivering, sweating
Precautions: mosquito nets, clean water
Medicines: chloroquine
Dosage: 1 tab twice a day
Emergency Signs: delirium or unconsciousness
Lifestyle Tips: Use mosquito nets, clean water

Disease: Typhoid
Description: A bacterial infection caused by Salmonella.
Symptoms: fever, abdominal pain, weakness
Precautions: boil water, avoid raw food
Medicines: azithromycin
Dosage: 1 tab once daily
Emergency Signs: severe dehydration
Lifestyle Tips: Boil drinking water, avoid raw food

Disease: Dengue
Description: A mosquito-borne viral disease.
Symptoms: fever, joint pain, rash
Precautions: rest, fluids
Medicines: paracetamol
Dosage: 1 tab thrice a day
Emergency Signs: internal bleeding
Lifestyle Tips: Avoid NSAIDs, rest & fluids

Disease: COVID-19
Description: A respiratory infection caused by the coronavirus.
Symptoms: fever, cough, loss of taste
Precautions: isolate, wear mask
Medicines: paracetamol, zinc
Dosage: 1 tab thrice a day
Emergency Signs: low oxygen
Lifestyle Tips: Isolate, wear mask, take vitamins

Disease: Asthma
Description: Airways inflamed and narrowed.
Symptoms: cough, short breath, wheezing
Precautions: avoid allergens
Medicines: albuterol inhaler
Dosage: 2 puffs as needed
Emergency Signs: cyanosis
Lifestyle Tips: Avoid allergens, use inhaler regularly

Disease: Diabetes
Description: High blood sugar levels.
Symptoms: fatigue, thirst, urination
Precautions: no sugar, check levels
Medicines: metformin
Dosage: 500mg twice daily
Emergency Signs: confusion or fainting
Lifestyle Tips: Monitor sugar, avoid sweets

Disease: Hypertension
Description: High blood pressure.
Symptoms: headache, dizziness, nosebleed
Precautions: low salt diet
Medicines: amlodipine
Dosage: 5mg once daily
Emergency Signs: blurred vision
Lifestyle Tips: Exercise, low-salt diet

Disease: Migraine
Description: Severe recurrent headache.
Symptoms: headache, nausea, light sensitivity
Precautions: avoid triggers
Medicines: sumatriptan
Dosage: 50mg as needed
Emergency Signs: vision loss
Lifestyle Tips: Avoid triggers, rest in dark

Disease: Pneumonia
Description: Infection inflaming the lungs.
Symptoms: cough, fever, short breath
Precautions: rest, antibiotics
Medicines: azithromycin
Dosage: 500mg once daily
Emergency Signs: blue lips or nails
Lifestyle Tips: Breathing exercises, fluids

Disease: TB
Description: Infectious lung disease.
Symptoms: cough, fever, weight loss
Precautions: mask, medication
Medicines: isoniazid
Dosage: 300mg daily
Emergency Signs: hemoptysis
Lifestyle Tips: Regular treatment, wear mask

Disease: Allergy
Description: Immune reaction to substances.
Symptoms: sneezing, rash, itching
Precautions: avoid allergens
Medicines: antihistamines
Dosage: 10mg once daily
Emergency Signs: anaphylaxis
Lifestyle Tips: Avoid allergens, use antihistamines

Disease: Bronchitis
Description: Inflammation of bronchial tubes.
Symptoms: cough, mucus, wheezing
Precautions: no smoke, hydration
Medicines: amoxicillin
Dosage: 500mg thrice daily
Emergency Signs: chest tightness
Lifestyle Tips: Avoid smoke, stay hydrated

Disease: Sinusitis
Description: Inflammation of the sinuses.
Symptoms: facial pain, nasal block, congestion
Precautions: steam, rest
Medicines: antibiotics
Dosage: 1 tab daily
Emergency Signs: fever with swelling
Lifestyle Tips: Use saline spray, rest

Disease: Conjunctivitis
Description: Inflammation of eye conjunctiva.
Symptoms: red eye, itchy eyes, tears
Precautions: no eye rubbing
Medicines: chloramphenicol drops
Dosage: 1 drop thrice daily
Emergency Signs: vision loss
Lifestyle Tips: Avoid touching eyes

Disease: UTI
Description: Bacterial infection of urinary tract.
Symptoms: burning urine, fever, urgency
Precautions: hydration, clean habits
Medicines: trimethoprim
Dosage: 100mg twice daily
Emergency Signs: flank pain
Lifestyle Tips: Hydrate well, avoid irritants

Disease: Anemia
Description: Low red blood cell count.
Symptoms: fatigue, pale skin, dizziness
Precautions: iron-rich food
Medicines: iron supplements
Dosage: 1 tab daily
Emergency Signs: chest pain
Lifestyle Tips: Iron-rich foods, rest

Disease: Gastritis
Description: Stomach lining inflammation.
Symptoms: nausea, bloating, stomach pain
Precautions: light meals
Medicines: omeprazole
Dosage: 20mg once daily
Emergency Signs: black stools
Lifestyle Tips: Light food, avoid spicy items

Disease: Acidity
Description: Excess acid in the stomach.
Symptoms: burning chest, sour belch, indigestion
Precautions: avoid late meals
Medicines: antacids
Dosage: 1 tab after food
Emergency Signs: severe chest pain
Lifestyle Tips: Avoid late meals, eat small portions

Disease: Jaundice
Description: Yellowing of skin/liver issue.
Symptoms: yellow eyes, fatigue, nausea
Precautions: avoid fatty food
Medicines: liver tonics
Dosage: as per dose
Emergency Signs: extreme weakness
Lifestyle Tips: Avoid oily food, hydrate well

Disease: Chickenpox
Description: Viral infection with rash.
Symptoms: rash, fever, itching
Precautions: rest, avoid scratching
Medicines: calamine, antivirals
Dosage: as per dose
Emergency Signs: high fever
Lifestyle Tips: Avoid scratching, rest

Disease: Measles
Description: Highly contagious virus.
Symptoms: fever, rash, runny nose
Precautions: isolate, hygiene
Medicines: vitamin A
Dosage: once daily
Emergency Signs: breathing issues
Lifestyle Tips: Isolate, good hygiene

Disease: Mumps
Description: Viral swelling of glands.
Symptoms: jaw pain, fever, swollen cheeks
Precautions: ice packs
Medicines: painkillers
Dosage: as needed
Emergency Signs: difficulty swallowing
Lifestyle Tips: Ice pack, avoid sour food

Disease: Scabies
Description: Skin infestation by mites.
Symptoms: itchy rash, burrows
Precautions: clean clothes
Medicines: permethrin cream
Dosage: apply overnight
Emergency Signs: secondary infection
Lifestyle Tips: Wash clothes, cut nails

Disease: Eczema
Description: Inflamed itchy skin.
Symptoms: red rash, itching, dryness
Precautions: avoid triggers
Medicines: moisturizer, steroids
Dosage: as per dose
Emergency Signs: severe inflammation
Lifestyle Tips: Avoid allergens

Disease: Psoriasis
Description: Autoimmune skin disorder.
Symptoms: scaly skin, red patches, itching
Precautions: moisturize, UV therapy
Medicines: steroid creams
Dosage: apply twice daily
Emergency Signs: joint pain
Lifestyle Tips: Use lotion, light therapy

Disease: Acne
Description: Blocked skin pores.
Symptoms: pimples, oily skin, scars
Precautions: wash face, avoid oil
Medicines: benzoyl peroxide
Dosage: apply daily
Emergency Signs: severe cysts
Lifestyle Tips: Avoid oily food, wash face

Disease: Depression
Description: Mood disorder.
Symptoms: sadness, low energy, sleep changes
Precautions: therapy, healthy food
Medicines: SSRIs
Dosage: as per doctor
Emergency Signs: suicidal thoughts
Lifestyle Tips: Talk therapy, regular schedule

Disease: Anxiety
Description: Feeling of fear/nervousness.
Symptoms: palpitations, sweating, worry
Precautions: meditation, no caffeine
Medicines: anxiolytics
Dosage: as per doctor
Emergency Signs: chest tightness
Lifestyle Tips: Meditation, reduce caffeine

Disease: Arthritis
Description: Joint inflammation.
Symptoms: joint pain, stiffness, swelling
Precautions: mild exercise
Medicines: NSAIDs
Dosage: as needed
Emergency Signs: limited motion
Lifestyle Tips: Hot pads, mild exercise

Disease: Back Pain
Description: Muscle or spine issues.
Symptoms: lower pain, stiffness, numbness
Precautions: stretch, posture
Medicines: painkillers
Dosage: as needed
Emergency Signs: leg weakness
Lifestyle Tips: Stretch, posture correction

Disease: Obesity
Description: Excess body fat.
Symptoms: weight gain, fatigue, breathlessness
Precautions: exercise, diet
Medicines: orlistat
Dosage: 120mg with meals
Emergency Signs: shortness of breath
Lifestyle Tips: Exercise, diet control

Disease: PCOS
Description: Hormonal disorder.
Symptoms: irregular periods, acne, weight gain
Precautions: healthy diet
Medicines: hormonal meds
Dosage: as per doctor
Emergency Signs: extreme pain
Lifestyle Tips: Healthy diet, weight management

Disease: Menstrual Cramps
Description: Pain during periods.
Symptoms: abdominal pain, nausea, fatigue
Precautions: heat pad, relax
Medicines: ibuprofen
Dosage: 400mg as needed
Emergency Signs: severe bleeding
Lifestyle Tips: Hot water bag, relax

Disease: Appendicitis
Description: Inflamed appendix.
Symptoms: abdominal pain, fever, nausea
Precautions: surgery required
Medicines: NA
Dosage: NA
Emergency Signs: rupture or shock
Lifestyle Tips: Immediate medical help

Disease: Tonsillitis
Description: Swollen tonsils.
Symptoms: sore throat, fever, difficulty swallowing
Precautions: lozenges, warm fluids
Medicines: penicillin
Dosage: 500mg daily
Emergency Signs: trouble breathing
Lifestyle Tips: Use lozenges, warm fluids

Disease: Vertigo
Description: Spinning sensation.
Symptoms: dizziness, nausea, balance issues
Precautions: get up slowly
Medicines: betahistine
Dosage: 16mg daily
Emergency Signs: fall risk
Lifestyle Tips: Change positions slowly

Disease: Ear Infection
Description: Middle ear inflammation.
Symptoms: earache, fever, hearing loss
Precautions: no objects in ear
Medicines: amoxicillin
Dosage: 500mg daily
Emergency Signs: ear discharge
Lifestyle Tips: Avoid inserting objects

Disease: Insomnia
Description: Difficulty sleeping.
Symptoms: sleeplessness, anxiety, tiredness
Precautions: no screens before bed
Medicines: melatonin
Dosage: 3mg at night
Emergency Signs: extreme fatigue
Lifestyle Tips: No caffeine, bedtime routine

Disease: GERD
Description: Acid reflux.
Symptoms: heartburn, bloating, belching
Precautions: no spicy food
Medicines: ranitidine
Dosage: 150mg daily
Emergency Signs: chest pain
Lifestyle Tips: Elevate head, avoid late meals

Disease: Lactose Intolerance
Description: Can't digest dairy.
Symptoms: bloating, diarrhea, gas
Precautions: avoid dairy
Medicines: lactase tablets
Dosage: as per use
Emergency Signs: extreme cramps
Lifestyle Tips: Avoid dairy, use alternatives
"""

# Convert the dataset to a dictionary for easy querying
def parse_dataset(dataset):
    diseases = {}
    lines = dataset.strip().split("\n\n")
    for entry in lines:
        details = entry.split("\n")
        disease_name = details[0].split(": ")[1]
        description = details[1].split(": ")[1]
        symptoms = details[2].split(": ")[1]
        precautions = details[3].split(": ")[1]
        medicines = details[4].split(": ")[1]
        dosage = details[5].split(": ")[1]
        emergency_signs = details[6].split(": ")[1]
        lifestyle_tips = details[7].split(": ")[1]

        diseases[disease_name] = {
            "description": description,
            "symptoms": symptoms,
            "precautions": precautions,
            "medicines": medicines,
            "dosage": dosage,
            "emergency_signs": emergency_signs,
            "lifestyle_tips": lifestyle_tips
        }
    
    return diseases

# Parse the dataset into a dictionary
disease_data = parse_dataset(dataset)

# Function to get disease recommendations
def get_disease_recommendation(disease_name):
    disease_info = disease_data.get(disease_name)
    if disease_info:
        return disease_info
    else:
        return "Disease not found in the database."

# Streamlit App Interface
st.title("Medicine Recommendation System")

st.sidebar.header("Choose Disease")
disease_list = list(disease_data.keys())
disease_choice = st.sidebar.selectbox("Select Disease", disease_list)

# Show disease information
if disease_choice:
    disease_info = get_disease_recommendation(disease_choice)
    st.subheader(f"Information for {disease_choice}")
    st.write(f"**Description**: {disease_info['description']}")
    st.write(f"**Symptoms**: {disease_info['symptoms']}")
    st.write(f"**Precautions**: {disease_info['precautions']}")
    st.write(f"**Medicines**: {disease_info['medicines']}")
    st.write(f"**Dosage**: {disease_info['dosage']}")
    st.write(f"**Emergency Signs**: {disease_info['emergency_signs']}")
    st.write(f"**Lifestyle Tips**: {disease_info['lifestyle_tips']}")

# Chatbot Functionality
st.sidebar.header("Chat with the AI")

user_input = st.text_input("Ask me anything about health:")

def chatbot_response(user_input):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose the model (GPT-3)
            prompt=user_input,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

if user_input:
    response = chatbot_response(user_input)
    st.subheader("Chatbot Response:")
    st.write(response)

# Streamlit secrets to manage API key securely

