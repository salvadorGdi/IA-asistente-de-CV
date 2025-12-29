import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuración inicial ---
st.set_page_config(page_title="Asistente de Análisis de Datasets", layout="centered")

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

# --- UI ---
st.title("📊 Asistente para Análisis de Datasets con IA")

st.write("""
Subí un archivo **CSV** y la IA te ayudará a entender:
- Qué contiene el dataset  
- Qué significan las columnas  
- Posibles usos y análisis  
""")

uploaded_file = st.file_uploader("📂 Subí tu archivo CSV", type=["csv"])

st.markdown("### ⚙️ ¿Cómo funciona?")
st.write("""
1. Subís un archivo CSV  
2. La app extrae información básica  
3. Un modelo de IA explica el dataset de forma clara  
""")

# --- Procesamiento ---
if uploaded_file and st.button("🔍 Analizar dataset"):
    try:
        df = pd.read_csv(uploaded_file)

        # 🔥 OPTIMIZACIÓN: solo info esencial
        info = f"""
Columnas: {list(df.columns)}
Tipos de datos: {df.dtypes.to_dict()}
Filas: {len(df)}
Valores faltantes (primeras 10 columnas):
{df.isnull().sum().head(10).to_dict()}
Ejemplo de registros:
{df.head(3).to_string()}
"""

        prompt = f"""
Actuá como un profesor de ciencia de datos.
Explicá este dataset de forma clara para un estudiante principiante.
No uses jerga técnica innecesaria.

Dataset:
{info}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sos un experto en análisis de datos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        st.success("✅ Análisis generado correctamente")
        st.markdown("### 🧠 Explicación de la IA")
        st.write(response.choices[0].message.content)

    except Exception as e:
        if "insufficient_quota" in str(e):
            st.warning("""
⚠️ La IA no pudo generar la explicación en este momento.

Esto ocurre porque la API utilizada requiere habilitar cuota de uso.
El código y el prompt funcionan correctamente, pero el proveedor
exige activación de billing para ejecución en producción.
""")
    else:
        st.error(f"Ocurrió un error inesperado: {e}")