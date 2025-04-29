import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from io import StringIO

# ------------------------
# Embedded CSV Dataset
# ------------------------
dataset = """
Disease,Description,Symptoms,Precautions,Medicines,Dosage,Emergency Signs,Lifestyle Tips
Common Cold,A viral infection of your nose and throat.,"cough,sneeze,congestion","rest,hydration",paracetamol,"1 tab thrice a day",severe chest pain,"Stay hydrated, rest well"
Flu,An infectious disease caused by the influenza virus.,"fever,cough,chills","rest,fluids","tamiflu,oseltamivir","1 tab twice a day",difficulty breathing,"Avoid cold drinks, wear warm clothes"
Malaria,A disease caused by a plasmodium parasite.,"fever,shivering,sweating","mosquito nets,clean water",chloroquine,"1 tab twice a day",delirium or unconsciousness,"Use mosquito nets, clean water"
Typhoid,A bacterial infection caused by Salmonella.,"fever,abdominal pain,weakness","boil water,avoid raw food",azithromycin,"1 tab once daily",severe dehydration,"Boil drinking water, avoid raw food"
Dengue,A mosquito-borne viral disease.,"fever,joint pain,rash","rest,fluids",paracetamol,"1 tab thrice a day",internal bleeding,"Avoid NSAIDs, rest & fluids"
COVID-19,A respiratory infection caused by the coronavirus.,"fever,cough,loss of taste","isolate,wear mask","paracetamol,zinc","1 tab thrice a day",low oxygen,"Isolate, wear mask, take vitamins"
Asthma,Airways inflamed and narrowed.,"cough,short breath,wheezing",avoid allergens,albuterol inhaler,"2 puffs as needed",cyanosis,"Avoid allergens, use inhaler regularly"
Diabetes,High blood sugar levels.,"fatigue,thirst,urination","no sugar,check levels",metformin,"500mg twice daily","confusion or fainting","Monitor sugar, avoid sweets"
Hypertension,High blood pressure.,"headache,dizziness,nosebleed","low salt diet",amlodipine,"5mg once daily",blurred vision,"Exercise, low-salt diet"
Migraine,Severe recurrent headache.,"headache,nausea,light sensitivity","avoid triggers",sumatriptan,"50mg as needed",vision loss,"Avoid triggers, rest in dark"
Pneumonia,Infection inflaming the lungs.,"cough,fever,short breath","rest,antibiotics",azithromycin,"500mg once daily","blue lips or nails","Breathing exercises, fluids"
TB,Infectious lung disease.,"cough,fever,weight loss","mask,medication",isoniazid,"300mg daily",hemoptysis,"Regular treatment, wear mask"
Allergy,Immune reaction to substances.,"sneezing,rash,itching",avoid allergens,antihistamines,"10mg once daily",anaphylaxis,"Avoid allergens, use antihistamines"
Bronchitis,Inflammation of bronchial tubes.,"cough,mucus,wheezing","no smoke,hydration",amoxicillin,"500mg thrice daily",chest tightness,"Avoid smoke, stay hydrated"
Sinusitis,Inflammation of the sinuses.,"facial pain,nasal block,congestion","steam,rest",antibiotics,"1 tab daily","fever with swelling","Use saline spray, rest"
Conjunctivitis,Inflammation of eye conjunctiva.,"red eye,itchy eyes,tears","no eye rubbing",chloramphenicol drops,"1 drop thrice daily",vision loss,"Avoid touching eyes"
UTI,Bacterial infection of urinary tract.,"burning urine,fever,urgency","hydration,clean habits",trimethoprim,"100mg twice daily",flank pain,"Hydrate well, avoid irritants"
Anemia,Low red blood cell count.,"fatigue,pale skin,dizziness","iron-rich food",iron supplements,"1 tab daily",chest pain,"Iron-rich foods, rest"
Gastritis,Stomach lining inflammation.,"nausea,bloating,stomach pain","light meals",omeprazole,"20mg once daily",black stools,"Light food, avoid spicy items"
Acidity,Excess acid in the stomach.,"burning chest,sour belch,indigestion","avoid late meals",antacids,"1 tab after food",severe chest pain,"Avoid late meals, eat small portions"
Jaundice,Yellowing of skin/liver issue.,"yellow eyes,fatigue,nausea","avoid fatty food",liver tonics,"as per dose",extreme weakness,"Avoid oily food, hydrate well"
Chickenpox,Viral infection with rash.,"rash,fever,itching","rest,avoid scratching","calamine,antivirals","as per dose",high fever,"Avoid scratching, rest"
Measles,Highly contagious virus.,"fever,rash,runny nose","isolate,hygiene",vitamin A,"once daily",breathing issues,"Isolate, good hygiene"
Mumps,Viral swelling of glands.,"jaw pain,fever,swollen cheeks",ice packs,painkillers,"as needed",difficulty swallowing,"Ice pack, avoid sour food"
Scabies,Skin infestation by mites.,"itchy rash,burrows",clean clothes,permethrin cream,"apply overnight",secondary infection,"Wash clothes, cut nails"
Eczema,Inflamed itchy skin.,"red rash,itching,dryness","avoid triggers","moisturizer,steroids","as per dose",severe inflammation,"Avoid allergens"
Psoriasis,Autoimmune skin disorder.,"scaly skin,red patches,itching","moisturize,UV therapy",steroid creams,"apply twice daily",joint pain,"Use lotion, light therapy"
Acne,Blocked skin pores.,"pimples,oily skin,scars","wash face,avoid oil",benzoyl peroxide,"apply daily",severe cysts,"Avoid oily food, wash face"
Depression,Mood disorder.,"sadness,low energy,sleep changes","therapy,healthy food",SSRIs,"as per doctor",suicidal thoughts,"Talk therapy, regular schedule"
Anxiety,Feeling of fear/nervousness.,"palpitations,sweating,worry","meditation,no caffeine",anxiolytics,"as per doctor",chest tightness,"Meditation, reduce caffeine"
Arthritis,Joint inflammation.,"joint pain,stiffness,swelling",mild exercise,NSAIDs,"as needed",limited motion,"Hot pads, mild exercise"
Back Pain,Muscle or spine issues.,"lower pain,stiffness,numbness","stretch,posture",painkillers,"as needed",leg weakness,"Stretch, posture correction"
Obesity,Excess body fat.,"weight gain,fatigue,breathlessness","exercise,diet",orlistat,"120mg with meals",shortness of breath,"Exercise, diet control"
PCOS,Hormonal disorder.,"irregular periods,acne,weight gain","healthy diet",hormonal meds,"as per doctor",extreme pain,"Healthy diet, weight management"
Menstrual Cramps,Pain during periods.,"abdominal pain,nausea,fatigue","heat pad,relax",ibuprofen,"400mg as needed",severe bleeding,"Hot water bag, relax"
Appendicitis,Inflamed appendix.,"abdominal pain,fever,nausea","surgery required",NA,NA,"rupture or shock","Immediate medical help"
Tonsillitis,Swollen tonsils.,"sore throat,fever,difficulty swallowing","lozenges,warm fluids",penicillin,"500mg daily",trouble breathing,"Use lozenges, warm fluids"
Vertigo,Spinning sensation.,"dizziness,nausea,balance issues","get up slowly",betahistine,"16mg daily",fall risk,"Change positions slowly"
Ear Infection,Middle ear inflammation.,"earache,fever,hearing loss","no objects in ear",amoxicillin,"500mg daily",ear discharge,"Avoid inserting objects"
Insomnia,Difficulty sleeping.,"sleeplessness,anxiety,tiredness","no screens before bed",melatonin,"3mg at night",extreme fatigue,"No caffeine, bedtime routine"
GERD,Acid reflux.,"heartburn,bloating,belching","no spicy food",ranitidine,"150mg daily",chest pain,"Elevate head, avoid late meals"
Lactose Intolerance,Can't digest dairy.,"bloating,diarrhea,gas",avoid dairy,lactase tablets,"as per use",extreme cramps,"Avoid dairy, use alternatives"
"""

# ------------------------
# Preprocessing
# ------------------------
df = pd.read_csv(StringIO(dataset))
df["Symptoms"] = df["Symptoms"].apply(lambda x: [s.strip() for s in x.split(",")])

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
y = df["Disease"]
model = RandomForestClassifier()
model.fit(X, y)

# ------------------------
# Streamlit UI Setup
# ------------------------
st.set_page_config("AI Health Advisor", layout="centered")
st.title("🧠 AI Health Advisor")
st.markdown("Your smart assistant for predicting disease and health guidance.")

health_tips = [
    "💧 Stay hydrated – drink 8+ glasses of water daily.",
    "😴 Aim for 7–8 hours of quality sleep every night.",
    "🧼 Wash hands frequently to avoid infections.",
    "🚶 Take a brisk 30-min walk daily for heart health.",
    "💊 Don’t self-medicate without expert advice.",
]
st.info(f"💡 Health Tip: **{random.choice(health_tips)}**")

# ------------------------
# Symptom Input
# ------------------------
all_symptoms = sorted(set(sym for sublist in df["Symptoms"] for sym in sublist))
selected = st.multiselect("🤒 Select your symptoms", all_symptoms)

if st.button("🔍 Predict Disease"):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        input_vec = mlb.transform([selected])
        result = model.predict(input_vec)[0]
        data = df[df["Disease"] == result].iloc[0]

        st.success(f"🩺 Predicted Disease: **{result}**")
        st.markdown(f"**About:** {data['Description']}")

        tab1, tab2, tab3, tab4 = st.tabs(["💊 Medicine", "🥗 Precautions", "⚠️ Emergency", "🌿 Lifestyle"])

        with tab1:
            st.write(f"**Medicines:** {data['Medicines']}")
            st.write(f"**Dosage:** {data['Dosage']}")

        with tab2:
            st.write(f"**Precautions:** {data['Precautions']}")

        with tab3:
            st.write(f"**Emergency Signs:** {data['Emergency Signs']}")

        with tab4:
            st.write(f"**Lifestyle Tips:** {data['Lifestyle Tips']}")



# ------------------------
# Offline AI Health Chatbot (No API Needed)
# ------------------------

st.header("🤖 Ask the Health Bot (Offline)")

user_question = st.text_area("💬 Ask anything about a disease, medicine, symptoms, etc.")

def keyword_matcher(question, df):
    question = question.lower()
    matched_rows = []

    # Try matching any disease name mentioned
    for _, row in df.iterrows():
        disease = row["Disease"].lower()
        if disease in question:
            matched_rows.append(row)

    if not matched_rows:
        return "🤔 Sorry, I couldn't find a matching disease. Please ask about a known condition."

    row = matched_rows[0]  # Only return the first match
    answer = ""

    # Smart reply builder
    if "medicine" in question:
        answer += f"💊 **Medicine for {row['Disease']}:** {row['Medicines']}\n\n"
        answer += f"📦 **Dosage:** {row['Dosage']}"
    elif "precaution" in question:
        answer += f"🛡️ **Precautions for {row['Disease']}:** {row['Precautions']}"
    elif "emergency" in question or "danger" in question:
        answer += f"⚠️ **Emergency Signs of {row['Disease']}:** {row['Emergency Signs']}"
    elif "symptom" in question:
        answer += f"🤒 **Symptoms of {row['Disease']}:** {', '.join(row['Symptoms'])}"
    elif "lifestyle" in question or "tip" in question:
        answer += f"🌿 **Lifestyle Tips for {row['Disease']}:** {row['Lifestyle Tips']}"
    elif "describe" in question or "about" in question:
        answer += f"🩺 **About {row['Disease']}:** {row['Description']}"
    else:
        answer += f"💡 Here's what I know about **{row['Disease']}**:\n\n"
        answer += f"- **Symptoms:** {', '.join(row['Symptoms'])}\n"
        answer += f"- **Precautions:** {row['Precautions']}\n"
        answer += f"- **Medicines:** {row['Medicines']} ({row['Dosage']})\n"
        answer += f"- **Emergency Signs:** {row['Emergency Signs']}\n"
        answer += f"- **Lifestyle Tips:** {row['Lifestyle Tips']}"

    return answer

if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        response = keyword_matcher(user_question, df)
        st.markdown(response)
