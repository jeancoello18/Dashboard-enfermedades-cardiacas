# Base del dashboard cardiovascular en Streamlit

## Importante
Esta carpeta contiene el punto de entrada de Streamlit y la configuración inicial.
**No incluye el modelo entrenado**, porque los archivos `scaler.joblib`, `modelo_final.joblib`,
`info_modelo.json`, `comparativa_examen.csv` e imágenes deben generarse desde el notebook
de Google Colab.

## 1. Generar artefactos en Colab
1. Cargar `processed.cleveland.data`.
2. Ejecutar todas las celdas del notebook.
3. Descargar `archivos_dashboard.zip`.
4. Descomprimirlo y copiar los archivos exportados a esta misma carpeta.

El notebook debe seleccionar Random Forest 2 si se mantienen los resultados y la regla
de selección descritos en el informe.

## 2. Revisar dependencias
El notebook genera un `requirements.txt` con la versión exacta de scikit-learn usada
para entrenar. Para evitar incompatibilidad al cargar el archivo Joblib, conserva esa
versión exacta y añade `plotly` y `streamlit` si no aparecen.

## 3. Ejecutar localmente (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app_examen.py
```

## 4. Publicar en Streamlit Community Cloud
1. Sube `app_examen.py`, `requirements.txt`, `.streamlit/config.toml` y todos los artefactos
   del modelo al repositorio GitHub.
2. En Streamlit Community Cloud selecciona el repositorio, la rama y `app_examen.py`.
3. Despliega y revisa los registros si aparece algún error.

## Seguridad y alcance
Los archivos Joblib deben proceder de una fuente confiable. No se deben cargar artefactos
desconocidos. Esta aplicación es académica y no constituye un dispositivo médico ni
proporciona diagnósticos clínicos.
