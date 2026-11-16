from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
st.set_page_config(
    page_title="Predicción de enfermedad cardiovascular",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "age", "sex", "chest_pain", "blood_pressure", "serum_cholestoral",
    "fasting_blood_sugar", "electrocardiographic", "max_heart_rate",
    "induced_angina", "ST_depression", "slope", "no_of_vessels", "thal"
]

FEATURE_LABELS = {
    "age": "Edad (años)",
    "sex": "Sexo codificado",
    "chest_pain": "Tipo de dolor torácico",
    "blood_pressure": "Presión arterial en reposo (mmHg)",
    "serum_cholestoral": "Colesterol sérico (mg/dl)",
    "fasting_blood_sugar": "Glucemia en ayunas > 120 mg/dl",
    "electrocardiographic": "Resultado electrocardiográfico en reposo",
    "max_heart_rate": "Frecuencia cardíaca máxima alcanzada",
    "induced_angina": "Angina inducida por ejercicio",
    "ST_depression": "Depresión ST inducida por ejercicio",
    "slope": "Pendiente del segmento ST",
    "no_of_vessels": "Número de vasos principales",
    "thal": "Resultado de talasemia (código)",
}

@st.cache_resource
def load_artifacts():
    model_path = BASE / "modelo_final.joblib"
    scaler_path = BASE / "scaler.joblib"
    info_path = BASE / "info_modelo.json"
    if not model_path.exists() or not scaler_path.exists() or not info_path.exists():
        raise FileNotFoundError(
            "Faltan modelo_final.joblib, scaler.joblib o info_modelo.json. "
            "Coloca todos los artefactos en la misma carpeta que app_examen.py."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return model, scaler, info

@st.cache_data
def load_comparison():
    path = BASE / "comparativa_examen.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def show_image(filename, caption):
    path = BASE / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"No se encontró el archivo {filename}.")

st.title("❤️ Predicción de enfermedad cardiovascular")
st.caption("Proyecto académico basado en el conjunto de datos Cleveland Heart Disease y metodología CRISP-ML(Q).")
st.warning(
    "Herramienta exclusivamente académica y demostrativa. No constituye un dispositivo médico, "
    "no diagnostica enfermedades y no sustituye la valoración de un profesional de salud."
)

try:
    model, scaler, info = load_artifacts()
except Exception as exc:
    st.error(f"No se pudieron cargar los artefactos: {exc}")
    st.stop()

comparison = load_comparison()
metricas = info.get("metricas_prueba", {})
model_name = info.get("nombre", "Modelo seleccionado")

with st.sidebar:
    st.header("Navegación")
    section = st.radio(
        "Selecciona una sección",
        ["Comparación de modelos", "Evaluación gráfica", "Predicción"],
    )
    st.divider()
    st.caption(f"Modelo de inferencia: **{model_name}**")
    st.caption("Dataset: Cleveland Heart Disease (303 registros originales).")

if section == "Comparación de modelos":
    st.subheader("Comparación de algoritmos")
    if comparison.empty:
        st.warning("No se encontró comparativa_examen.csv.")
    else:
        subset = st.selectbox("Conjunto de evaluación", ["Prueba", "Entrenamiento"])
        df = comparison[comparison["Conjunto"] == subset].copy()
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("#### Comparación de métricas")
        metric = st.selectbox(
            "Métrica para comparar",
            ["Exactitud", "Precisión", "Recall", "Especificidad", "F1", "AUC"],
        )
        chart_df = df.set_index("Algoritmo")[[metric]]
        st.bar_chart(chart_df)
        st.caption("Los valores se presentan tal como aparecen en comparativa_examen.csv.")

elif section == "Evaluación gráfica":
    st.subheader("Evaluación gráfica de los modelos")
    st.markdown("#### Curvas ROC")
    show_image("roc_curvas.png", "Curvas ROC de los algoritmos evaluados")
    st.markdown("#### Matrices de confusión")
    show_image("matrices_confusion.png", "Matrices de confusión")
    st.markdown("#### Curvas de aprendizaje y pérdida")
    show_image("curvas_aprendizaje.png", "Curvas de rendimiento y pérdida del modelo MLP")

elif section == "Predicción":
    st.subheader("Evaluar un nuevo registro")
    st.write(
        "Introduce los valores de las variables. Los códigos categóricos corresponden "
        "a la codificación del conjunto Cleveland utilizado en el experimento."
    )
    with st.form("patient_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Edad (años)", min_value=1, max_value=120, value=55)
            sex = st.selectbox("Sexo codificado", [0, 1], format_func=lambda x: "0" if x == 0 else "1")
            chest_pain = st.selectbox("Tipo de dolor torácico", [1, 2, 3, 4])
            blood_pressure = st.number_input("Presión arterial en reposo (mmHg)", min_value=0.0, max_value=300.0, value=130.0)
            serum_cholestoral = st.number_input("Colesterol sérico (mg/dl)", min_value=0.0, max_value=800.0, value=240.0)
            fasting_blood_sugar = st.selectbox("Glucemia en ayunas > 120 mg/dl", [0, 1])
            electrocardiographic = st.selectbox("Resultado electrocardiográfico en reposo", [0, 1, 2])
        with right:
            max_heart_rate = st.number_input("Frecuencia cardíaca máxima alcanzada", min_value=0.0, max_value=300.0, value=150.0)
            induced_angina = st.selectbox("Angina inducida por ejercicio", [0, 1])
            ST_depression = st.number_input("Depresión ST inducida por ejercicio", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Pendiente del segmento ST", [1, 2, 3])
            no_of_vessels = st.selectbox("Número de vasos principales", [0, 1, 2, 3])
            thal = st.selectbox("Resultado de talasemia (código)", [3, 6, 7])
        submitted = st.form_submit_button("Generar predicción", type="primary", use_container_width=True)

    if submitted:
        values = {
            "age": age,
            "sex": sex,
            "chest_pain": chest_pain,
            "blood_pressure": blood_pressure,
            "serum_cholestoral": serum_cholestoral,
            "fasting_blood_sugar": fasting_blood_sugar,
            "electrocardiographic": electrocardiographic,
            "max_heart_rate": max_heart_rate,
            "induced_angina": induced_angina,
            "ST_depression": ST_depression,
            "slope": slope,
            "no_of_vessels": no_of_vessels,
            "thal": thal,
        }
        patient = pd.DataFrame([values], columns=FEATURES)
        try:
            scaled = scaler.transform(patient)
            prediction = int(model.predict(scaled)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(scaled)[0][1])
            st.divider()
            if prediction == 1:
                st.error("Resultado del modelo: clase 1 — el modelo estima presencia de enfermedad.")
            else:
                st.success("Resultado del modelo: clase 0 — el modelo estima ausencia de enfermedad.")
            if probability is not None:
                st.metric("Probabilidad estimada para la clase 1", f"{probability:.1%}")
            st.caption("La salida es una predicción estadística del modelo, no un diagnóstico clínico.")
        except Exception as exc:
            st.error(f"No fue posible generar la predicción: {exc}")


st.divider()
st.caption("Proyecto académico — Cleveland Heart Disease. Las predicciones no deben utilizarse para decisiones médicas.")
