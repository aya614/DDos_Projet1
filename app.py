import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Détection & Prédiction d'Attaques DDoS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; color: #1E88E5; font-weight: bold; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #555555; margin-bottom: 25px; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Authentication Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>🛡️ Portail Sécurisé DDoS Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Veuillez vous authentifier pour accéder au tableau de bord.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            submit = st.form_submit_button("Se connecter", use_container_width=True)
            if submit:
                if username == "ayaesslt" and password == "aya123":
                    st.session_state.authenticated = True
                    st.success("Connexion réussie !")
                    st.rerun()
                else:
                    st.error("Identifiants incorrects.")

if not st.session_state.authenticated:
    login()
    st.stop()

# Logout in sidebar
with st.sidebar:
    st.title("👤 Session")
    st.write("Connecté en tant que **ayaesslt**")
    if st.button("Se déconnecter", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()

# Load & Cache Data & Training
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("dataset.csv")
    df_clean = df.drop_duplicates().fillna(0)
    
    y = df_clean["label"]
    X = df_clean.drop(columns=["label", "timestamp", "src_ip", "dst_ip", "src_port"], errors="ignore")
    
    # One-hot encoding
    cat_cols = [c for c in ["protocol", "dst_port"] if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols, dtype=int)
    
    return df, df_clean, X_encoded, y

df_raw, df_clean, X, y = load_and_preprocess_data()

@st.cache_resource
def train_models(X_train_df, y_train_data, X_val_df, y_val_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    
    models = {
        "Régression linéaire": LinearRegression(),
        "Régression logistique": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1)
    }
    
    preds = {}
    metrics = []
    
    for name, model in models.items():
        if name in ["Random Forest", "XGBoost"]:
            model.fit(X_train_df, y_train_data)
            y_pred = model.predict(X_val_df)
        else:
            model.fit(X_train_scaled, y_train_data)
            pred_raw = model.predict(X_val_scaled)
            y_pred = (pred_raw >= 0.5).astype(int) if name == "Régression linéaire" else pred_raw
            
        preds[name] = y_pred
        metrics.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_val_data, y_pred),
            "Précision": precision_score(y_val_data, y_pred, zero_division=0),
            "Recall": recall_score(y_val_data, y_pred, zero_division=0),
            "F1-score": f1_score(y_val_data, y_pred, zero_division=0)
        })
        
    metrics_df = pd.DataFrame(metrics).sort_values("F1-score", ascending=False)
    return models, scaler, preds, metrics_df

# Train/Val Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

models, scaler, val_preds, metrics_df = train_models(X_train, y_train, X_val, y_val)

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Aller à", [
    "📊 Statistiques & Explorer",
    "🤖 Comparaison des Modèles",
    "⚡ Prédiction en direct"
])

# Page 1: Stats & Exploration
if page == "📊 Statistiques & Explorer":
    st.markdown("<p class='main-title'>📊 Exploration du Dataset DDoS</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Analyse descriptive et répartition des flux réseau</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Logs", f"{len(df_raw):,}")
    col2.metric("Trafic Normal (0)", f"{(y == 0).sum():,} ({((y == 0).mean()*100):.1f}%)")
    col3.metric("Trafic DDoS (1)", f"{(y == 1).sum():,} ({((y == 1).mean()*100):.1f}%)")
    col4.metric("Caractéristiques", f"{X.shape[1]}")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Répartition des Classes")
        fig_pie = px.pie(
            names=["Normal (0)", "DDoS (1)"],
            values=[(y == 0).sum(), (y == 1).sum()],
            color_discrete_sequence=["#2ECC71", "#E74C3C"],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.subheader("Distribution du Protocole par Classe")
        if "protocol" in df_clean.columns:
            fig_bar = px.histogram(
                df_clean, x="protocol", color="label", barmode="group",
                color_discrete_map={0: "#2ECC71", 1: "#E74C3C"},
                labels={"label": "Classe (0: Normal, 1: DDoS)"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
    st.subheader("Aperçu des Données Réseau Cleaned")
    st.dataframe(df_clean.head(10), use_container_width=True)

# Page 2: Model Comparison
elif page == "🤖 Comparaison des Modèles":
    st.markdown("<p class='main-title'>🤖 Performances des Algorithmes Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Évaluation sur l'ensemble de validation (15% dataset)</p>", unsafe_allow_html=True)
    
    st.subheader("Tableau Comparatif des Métriques")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#D4EFDF"), use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Comparaison du F1-Score")
        fig_f1 = px.bar(
            metrics_df, x="Modèle", y="F1-score", color="F1-score",
            color_continuous_scale="Blues", text_auto=".3f"
        )
        fig_f1.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_f1, use_container_width=True)
        
    with col2:
        selected_model = st.selectbox("Choisir un modèle pour la Matrice de Confusion", list(models.keys()), index=4)
        st.subheader(f"Matrice de Confusion — {selected_model}")
        
        y_pred_sel = val_preds[selected_model]
        cm = confusion_matrix(y_val, y_pred_sel)
        
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Prédit Normal", "Prédit DDoS"],
            y=["Réel Normal", "Réel DDoS"],
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# Page 3: Live Prediction
elif page == "⚡ Prédiction en direct":
    st.markdown("<p class='main-title'>⚡ Détection DDoS en Temps Réel</p>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Saisissez les paramètres de flux pour prédire la nature du trafic</p>", unsafe_allow_html=True)
    
    chosen_model_name = st.selectbox("Modèle de classification", ["XGBoost", "Random Forest", "SVM", "Régression logistique"], index=0)
    model = models[chosen_model_name]
    
    st.divider()
    
    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            protocol = st.selectbox("Protocole", ["TCP", "UDP", "ICMP"])
            dst_port = st.selectbox("Port Destination", [80, 443, 22, 53, 8080])
            duration = st.number_input("Durée du flux (s)", min_value=0.0, max_value=60.0, value=2.5)
            pkt_count = st.number_input("Nombre de paquets (pkt_count)", min_value=1, max_value=2000, value=350)
        with col2:
            byte_count = st.number_input("Taille totale (octets)", min_value=64, max_value=5000000, value=450000)
            flow_rate = st.number_input("Débit (flow_rate)", min_value=0.0, max_value=1000000.0, value=120000.0)
            syn_count = st.number_input("SYN Count", min_value=0, max_value=500, value=40)
            ack_count = st.number_input("ACK Count", min_value=0, max_value=500, value=5)
        with col3:
            rst_count = st.number_input("RST Count", min_value=0, max_value=100, value=1)
            fin_count = st.number_input("FIN Count", min_value=0, max_value=100, value=0)
            psh_count = st.number_input("PSH Count", min_value=0, max_value=100, value=2)
            urg_count = st.number_input("URG Count", min_value=0, max_value=50, value=0)
            avg_pkt_size = st.number_input("Taille moyenne paquet", min_value=0.0, max_value=2000.0, value=1285.7)
            
        submit_pred = st.form_submit_button("🔍 Analyser le flux", use_container_width=True)
        
    if submit_pred:
        # Build vector matching X columns
        input_dict = {col: 0 for col in X.columns}
        
        # Numeric features
        input_dict["duration"] = duration
        input_dict["pkt_count"] = pkt_count
        input_dict["byte_count"] = byte_count
        input_dict["flow_rate"] = flow_rate
        input_dict["syn_count"] = syn_count
        input_dict["ack_count"] = ack_count
        input_dict["rst_count"] = rst_count
        input_dict["fin_count"] = fin_count
        input_dict["psh_count"] = psh_count
        input_dict["urg_count"] = urg_count
        input_dict["avg_pkt_size"] = avg_pkt_size
        
        # Categorical one-hot features
        proto_col = f"protocol_{protocol}"
        port_col = f"dst_port_{dst_port}"
        if proto_col in input_dict:
            input_dict[proto_col] = 1
        if port_col in input_dict:
            input_dict[port_col] = 1
            
        input_df = pd.DataFrame([input_dict])
        
        if chosen_model_name in ["Random Forest", "XGBoost"]:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
        else:
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
            
        st.divider()
        if pred == 1:
            st.error(f"🚨 **ALERTE DDOS DÉTECTÉE !** (Probabilité d'attaque: {prob*100:.1f}%)")
            st.warning("Recommandation: Déclencher les règles de filtrage de pare-feu et la limitation de débit sur l'IP source.")
        else:
            st.success(f"✅ **TRAFIC NORMAL DETECTÉ** (Confiance normal: {(1-prob)*100:.1f}%)")
            st.info("Le flux analysé ne présente pas d'anomalie ni de comportement d'inondation.")
