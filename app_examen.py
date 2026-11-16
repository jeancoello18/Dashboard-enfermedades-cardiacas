from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Predicción Cardiovascular | Grupo 4",
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
    "chest_pain": "Tipo de dolor de pecho",
    "blood_pressure": "Presión arterial en reposo (mmHg)",
    "serum_cholestoral": "Colesterol sérico (mg/dl)",
    "fasting_blood_sugar": "Azúcar en ayunas > 120 mg/dl",
    "electrocardiographic": "Resultado electrocardiográfico",
    "max_heart_rate": "Frecuencia cardíaca máxima",
    "induced_angina": "Angina inducida por ejercicio",
    "ST_depression": "Depresión ST",
    "slope": "Pendiente ST",
    "no_of_vessels": "Número de vasos principales",
    "thal": "Thal"
}

@st.cache_resource
def load_artifacts():
    scaler_path = BASE_DIR / "scaler.joblib"
    model_path = BASE_DIR / "modelo_final.joblib"
    info_path = BASE_DIR / "info_modelo.json"

    missing = [p.name for p in [scaler_path, model_path, info_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos exportados desde Google Colab: " + ", ".join(missing)
        )

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return scaler, model, info

def load_comparison():
    path = BASE_DIR / "comparativa_examen.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def show_image(filename, caption):
    path = BASE_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"La imagen **{filename}** todavía no está en el repositorio. Vuelve a ejecutar la celda de exportación de Google Colab.")

st.title("❤️ Dashboard de Predicción de Enfermedad Cardiovascular")
st.caption("Proyecto académico · Metodología CRISP-ML(Q) · Grupo 4")
st.warning(
    "Herramienta educativa y experimental. Sus resultados no constituyen un diagnóstico "
    "médico ni sustituyen la valoración de un profesional de salud."
)

try:
    scaler, model, info = load_artifacts()
except Exception as exc:
    st.error(f"No se pudieron cargar los artefactos del modelo. {exc}")
    st.markdown(
        """
        **Antes de publicar:** ejecuta todas las celdas del notebook en Google Colab,
        descarga `archivos_dashboard.zip`, descomprímelo y copia los archivos generados
        junto a `app_examen.py` en este repositorio.
        """
    )
    st.stop()

model_name = info.get("nombre", "Modelo no identificado")
model_type = info.get("tipo", "")
metrics = info.get("metricas_prueba", {})

if model_type != "sklearn":
    st.error(
        "El artefacto exportado corresponde a una red neuronal o no es compatible "
        "con esta versión de la aplicación. Este dashboard está preparado para "
        "Random Forest 2, seleccionado en el informe."
    )
    st.stop()

if model_name != "Random Forest 2":
    st.warning(
        f"El archivo indica que el modelo ganador es **{model_name}**, no Random Forest 2. "
        "Verifica que la ejecución de Colab y el informe utilicen la misma selección."
    )

# Resumen de métricas del modelo ganador
st.subheader("Resumen del modelo seleccionado")
kpi_cols = st.columns(4)
for col, (label, key) in zip(
    kpi_cols,
    [("F1", "F1"), ("Recall", "Recall"), ("AUC", "AUC"), ("Exactitud", "Exactitud")]
):
    value = metrics.get(key)
    col.metric(label, f"{float(value):.3f}" if value is not None else "N/D")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Inicio", "Comparación", "Evaluación", "Predicción", "CRISP-ML(Q)"
])

with tab1:
    st.header("Resumen del proyecto")
    st.write(
        "El sistema compara tres algoritmos de aprendizaje automático y una red neuronal "
        "MLP para clasificar el diagnóstico cardiovascular del conjunto Cleveland de UCI."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes del conjunto", "303")
    c2.metric("Variables predictoras", "13")
    c3.metric("Modelo seleccionado", model_name)
    st.markdown(
        """
        **Criterio de selección:** mayor F1 en el conjunto de prueba; en caso de empate,
        mayor Recall y, si persiste, mayor AUC. La clase positiva representa pacientes
        etiquetados como enfermos en el conjunto de datos.
        """
    )
    st.info(
        "La selección reportada en el informe es Random Forest 2: F1 = 0,868, "
        "Recall = 0,846 y AUC = 0,927. Los indicadores superiores se leen del "
        "archivo `info_modelo.json` generado por la última ejecución del notebook."
    )

with tab2:
    st.header("Comparación de algoritmos")
    comparison = load_comparison()
    if comparison is None:
        st.warning("No se encontró comparativa_examen.csv. Ejecuta la exportación del notebook.")
    else:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        test_data = comparison[comparison["Conjunto"].str.lower() == "prueba"].copy()
        metric_options = [m for m in ["Exactitud", "Precisión", "Recall", "Especificidad", "F1", "AUC"]
                          if m in test_data.columns]
        if not test_data.empty and metric_options:
            selected_metric = st.selectbox("Métrica para comparar", metric_options, index=metric_options.index("F1") if "F1" in metric_options else 0)
            fig = px.bar(
                test_data, x="Algoritmo", y=selected_metric, color="Tipo",
                title=f"{selected_metric} por algoritmo · conjunto de prueba",
                text_auto=".3f", range_y=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Evaluación de los modelos")
    st.subheader("Curvas ROC")
    show_image("roc_curvas.png", "Curvas ROC de los cuatro algoritmos en entrenamiento y prueba")
    st.subheader("Matrices de confusión")
    show_image("matrices_confusion.png", "Matrices de confusión en el conjunto de prueba")
    st.subheader("Aprendizaje de la red neuronal MLP")
    show_image("curvas_aprendizaje.png", "Exactitud y pérdida durante entrenamiento y validación")

with tab4:
    st.header("Predicción para un nuevo paciente")
    st.write(
        "Introduce los valores de las variables clínicas. Los campos categóricos conservan "
        "los códigos utilizados por el dataset Cleveland."
    )

    with st.form("patient_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Edad (años)", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sexo", options=[(1, "Masculino (1)"), (0, "Femenino (0)")], format_func=lambda x: x[1])[0]
            chest_pain = st.selectbox(
                "Tipo de dolor de pecho",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1:"Angina típica (1)", 2:"Angina atípica (2)", 3:"Dolor no anginoso (3)", 4:"Asintomático (4)"}[x]
            )
            blood_pressure = st.number_input("Presión arterial en reposo (mmHg)", min_value=50, max_value=250, value=130)
            serum_cholestoral = st.number_input("Colesterol sérico (mg/dl)", min_value=50, max_value=700, value=240)
            fasting_blood_sugar = st.selectbox("Azúcar en ayunas > 120 mg/dl", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Sí (1)")
            electrocardiographic = st.selectbox("Resultado electrocardiográfico", options=[0, 1, 2])
        with right:
            max_heart_rate = st.number_input("Frecuencia cardíaca máxima", min_value=50, max_value=250, value=150)
            induced_angina = st.selectbox("Angina inducida por ejercicio", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Sí (1)")
            ST_depression = st.number_input("Depresión ST", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Pendiente ST", options=[1, 2, 3], format_func=lambda x: {1:"Ascendente (1)", 2:"Plana (2)", 3:"Descendente (3)"}[x])
            no_of_vessels = st.selectbox("Número de vasos principales", options=[0, 1, 2, 3])
            thal = st.selectbox("Thal", options=[3, 6, 7], format_func=lambda x: {3:"Normal (3)", 6:"Defecto fijo (6)", 7:"Defecto reversible (7)"}[x])

        submitted = st.form_submit_button("Generar predicción", type="primary", use_container_width=True)

    if submitted:
        patient = pd.DataFrame([[
            age, sex, chest_pain, blood_pressure, serum_cholestoral,
            fasting_blood_sugar, electrocardiographic, max_heart_rate,
            induced_angina, ST_depression, slope, no_of_vessels, thal
        ]], columns=FEATURES)

        try:
            patient_scaled = scaler.transform(patient)
            prediction = int(model.predict(patient_scaled)[0])
            probability = (
                float(model.predict_proba(patient_scaled)[0][1])
                if hasattr(model, "predict_proba") else None
            )

            st.divider()
            if prediction == 1:
                st.error("Resultado del modelo: clase 1 · Enfermedad presente según la etiqueta del dataset")
            else:
                st.success("Resultado del modelo: clase 0 · Sin enfermedad según la etiqueta del dataset")

            if probability is not None:
                st.metric("Probabilidad estimada por el modelo para la clase 1", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
            st.caption(
                "La probabilidad es una salida del modelo entrenado; no equivale por sí sola "
                "a un riesgo clínico individual ni confirma o descarta una enfermedad."
            )
            with st.expander("Ver datos enviados al modelo"):
                st.dataframe(patient.rename(columns=FEATURE_LABELS), use_container_width=True)
        except Exception as exc:
            st.error(f"No fue posible generar la predicción: {exc}")

with tab5:
    st.header("Metodología CRISP-ML(Q)")
    phases = [
        ("1. Comprensión del negocio y de los datos", "Predicción binaria de enfermedad cardíaca con el conjunto Cleveland UCI: 303 pacientes y 13 variables predictoras."),
        ("2. Preparación de datos", "Tratamiento de valores faltantes con la moda, binarización de diagnosis y estandarización con StandardScaler."),
        ("3. Modelado", "Random Forest 1, Random Forest 2 con búsqueda GridSearchCV, Regresión Logística y red neuronal MLP."),
        ("4. Evaluación", "Comparación de exactitud, precisión, Recall, especificidad, F1, AUC, curvas ROC y matrices de confusión."),
        ("5. Despliegue", "Carga del modelo ganador y del escalador guardados desde Google Colab para predecir registros nuevos."),
        ("6. Monitoreo y mantenimiento", "Propuesta: registrar versiones, revisar periódicamente métricas con datos nuevos etiquetados, detectar cambios en los datos y reentrenar solo tras una evaluación documentada.")
    ]
    for title, description in phases:
        with st.expander(title, expanded=False):
            st.write(description)
    st.caption(
        "La fase de monitoreo mostrada es una propuesta de trabajo; no representa un "
        "monitoreo clínico ni un proceso automático ya implementado."
    )

st.divider()
st.caption("Universidad Técnica de Manabí · Grupo 4 · Proyecto académico de clasificación cardiovascular")
