import streamlit as st
import pandas as pd
import joblib

# ======================
# Load Model
# ======================

from pathlib import Path


MODEL_PATH = Path(__file__).parent / "dependency_model.pkl"

model = joblib.load(MODEL_PATH)

# ======================
# Page Config
# ======================

st.set_page_config(
    page_title="AI Dependency Predictor",
    page_icon="🤖",
    layout="centered"
)

# ======================
# Sidebar
# ======================

with st.sidebar:
    st.header("📊 Información del Modelo")

    st.metric("Datos de entrenamiento", "112")
    st.metric("Accuracy", "0.86")
    st.metric("Modelo", "CategoricalNB")

    st.markdown("---")

    st.write(
        """
        Este modelo fue entrenado utilizando respuestas de una encuesta
        sobre dependencia emocional y conductual hacia sistemas de IA.
        """
    )

# ======================
# Title
# ======================

st.title("🤖 Predictor de Dependencia hacia la IA")

st.markdown("""
Esta aplicación utiliza un modelo **Categorical Naive Bayes**
para estimar el nivel de dependencia hacia sistemas de Inteligencia Artificial.

Las categorías posibles son:

- 🟢 Bajo
- 🟡 Medio
- 🔴 Alto
""")

st.divider()

# ======================
# Inputs
# ======================

ai_hours = st.selectbox(
    "¿Cuántas horas al día utilizas IA?",
    [
        "Menos de 1 hora",
        "1–2 horas",
        "2–4 horas",
        "Más de 4 horas"
    ]
)

social_interaction = st.selectbox(
    "¿Con qué frecuencia interactúas socialmente en persona?",
    [
        "Rara vez",
        "Ocasionalmente",
        "Frecuentemente",
        "Muy frecuentemente"
    ]
)

ai_frequency = st.selectbox(
    "¿Con qué frecuencia utilizas herramientas de IA?",
    [
        "Rara vez",
        "Una vez al día",
        "Varias veces al día",
        "Algunas veces a la semana"
    ]
)

emotional_support = st.radio(
    "¿Has utilizado IA para apoyo emocional?",
    ["Sí", "No"]
)

follows_advice = st.radio(
    "¿Sueles seguir consejos proporcionados por la IA?",
    ["Sí", "No"]
)

help_source = st.selectbox(
    "Cuando tienes un problema, ¿a quién recurres principalmente?",
    [
        "Ambos por igual",
        "IA",
        "Personas (amigos, familia)"
    ]
)

usage_type = st.selectbox(
    "¿Cuál es tu uso principal de la IA?",
    [
        "Apoyo emocional / personal",
        "Entretenimiento",
        "Estudio / tareas académicas",
        "Trabajo"
    ]
)

# ======================
# Encodings
# ======================

ai_hours_map = {
    "Menos de 1 hora": 0,
    "1–2 horas": 1,
    "2–4 horas": 2,
    "Más de 4 horas": 3
}

social_map = {
    "Rara vez": 0,
    "Ocasionalmente": 1,
    "Frecuentemente": 2,
    "Muy frecuentemente": 3
}

frequency_map = {
    "Rara vez": 0,
    "Una vez al día": 1,
    "Varias veces al día": 2,
    "Algunas veces a la semana": 3
}

help_map = {
    "Ambos por igual": 0,
    "IA": 1,
    "Personas (amigos, familia)": 2
}

usage_map = {
    "Apoyo emocional / personal": 0,
    "Entretenimiento": 1,
    "Estudio / tareas académicas": 2,
    "Trabajo": 3
}

# ======================
# Prediction
# ======================

if st.button("🔍 Predecir nivel de dependencia"):

    X = pd.DataFrame([[
        ai_hours_map[ai_hours],
        social_map[social_interaction],
        frequency_map[ai_frequency],
        1 if emotional_support == "Sí" else 0,
        1 if follows_advice == "Sí" else 0,
        help_map[help_source],
        usage_map[usage_type]
    ]])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    labels = {
        0: "🟢 Bajo",
        1: "🟡 Medio",
        2: "🔴 Alto"
    }

    st.success(
        f"Nivel de dependencia estimado: {labels[prediction]}"
    )

    # ======================
    # Probability Chart
    # ======================

    st.subheader("📈 Distribución de Probabilidades")

    prob_df = pd.DataFrame({
        "Nivel": ["Bajo", "Medio", "Alto"],
        "Probabilidad": probabilities
    })

    st.bar_chart(
        prob_df.set_index("Nivel")
    )

    st.write(
        f"🟢 Bajo: {probabilities[0]:.1%}"
    )

    st.write(
        f"🟡 Medio: {probabilities[1]:.1%}"
    )

    st.write(
        f"🔴 Alto: {probabilities[2]:.1%}"
    )

    st.divider()

    # ======================
    # Interpretation
    # ======================

    st.subheader("🧠 Interpretación")

    if prediction == 0:
        st.info(
            "El modelo identifica un patrón consistente con baja dependencia hacia sistemas de inteligencia artificial."
        )

    elif prediction == 1:
        st.warning(
            "El modelo identifica señales moderadas de dependencia y uso frecuente de herramientas de IA."
        )

    else:
        st.error(
            "El modelo identifica características asociadas con una mayor dependencia hacia sistemas de inteligencia artificial."
        )

st.divider()

st.caption(
    "Proyecto académico sobre dependencia hacia la IA utilizando métodos estadísticos y aprendizaje automático."
)