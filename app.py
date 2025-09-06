import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pennylane as qml
import pennylane.numpy as pnp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# --- Page setup ---
st.set_page_config(page_title="Quantum Forecasting System", layout="wide")
st.title("🔮 Quantum Forecasting System")
st.markdown("Compare **Quantum vs Classical forecasting** with interactive plots and Bloch sphere visualization.")

# --- Sidebar controls ---
st.sidebar.header("⚙️ Settings")
model_type = st.sidebar.radio("Select Model", ["Quantum", "Classical"])
layers = st.sidebar.slider("Quantum Layers", 1, 5, 2)
epochs = st.sidebar.slider("Training Epochs", 10, 200, 50, 10)
lr = st.sidebar.select_slider("Learning Rate", options=[0.1, 0.05, 0.01, 0.005, 0.001], value=0.05)

data_choice = st.sidebar.radio("Dataset", ["Sample Sine Data", "Upload CSV"])

# --- Data preparation ---
if data_choice == "Sample Sine Data":
    n_samples = 60
    X = np.linspace(0, 2*np.pi, n_samples)
    y_true = np.sin(2*X) + 0.3*np.sin(5*X) + 0.1*X + 0.1*np.random.randn(n_samples)
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV with one numeric column", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col = st.sidebar.selectbox("Select column", df.columns)
        y_true = df[col].values
        X = np.arange(len(y_true))
    else:
        st.warning("Upload a CSV file to continue.")
        st.stop()

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_true[:split], y_true[split:]

# --- Quantum model ---
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def circuit(x, weights):
    qml.RX(x, wires=0)
    for W in weights:
        qml.Rot(W[0], W[1], W[2], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def circuit_state(x, weights):
    qml.RX(x, wires=0)
    for W in weights:
        qml.Rot(W[0], W[1], W[2], wires=0)
    return qml.state()

def quantum_forecast(X_train, y_train, X_test):
    # Initialize weights compatible with autograd
    weights = pnp.array(np.random.randn(layers, 3), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    # Cost function compatible with autograd
    def cost(w):
        preds = pnp.zeros(len(X_train), dtype=float)
        for i, x in enumerate(X_train):
            preds[i] = pnp.array(circuit(x, w))  # ensures ArrayBox compatibility
        return pnp.mean((preds - y_train)**2)

    start = time.time()
    for _ in range(epochs):
        weights = opt.step(cost, weights)
    end = time.time()

    # Predictions for plotting (outside gradient loop)
    preds = np.array([circuit(x, weights).item() for x in X_test])
    return preds, end-start, weights

# --- Classical model ---
def classical_forecast(X_train, y_train, X_test):
    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X_train.reshape(-1, 1))
    model = LinearRegression()
    start = time.time()
    model.fit(X_poly, y_train)
    end = time.time()
    preds = model.predict(poly.transform(X_test.reshape(-1, 1)))
    return preds, end-start

# --- Run model ---
if st.sidebar.button("🚀 Run Model"):
    if model_type == "Quantum":
        y_pred, t, weights = quantum_forecast(X_train, y_train, X_test)
    else:
        y_pred, t = classical_forecast(X_train, y_train, X_test)
        weights = None

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.success(f"✅ {model_type} Model finished in {t:.2f}s | MAE={mae:.3f}, RMSE={rmse:.3f}")

    # --- Forecast plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_true, mode="lines", name="True"))
    fig.add_trace(go.Scatter(x=X_test, y=y_pred, mode="lines+markers", name=f"{model_type} Forecast"))
    st.plotly_chart(fig, use_container_width=True)

    # --- Bloch Sphere (Quantum only) ---
    if model_type == "Quantum" and weights is not None:
        bloch_points = []
        for x in X_test:
            state = circuit_state(x, weights)
            bloch_points.append([
                2*np.real(np.vdot([1,0], state)*np.vdot(state,[0,1])),  # X
                2*np.imag(np.vdot([1,0], state)*np.vdot(state,[0,1])),  # Y
                np.abs(state[0])**2 - np.abs(state[1])**2               # Z
            ])

        bloch_points = np.array(bloch_points)
        bloch_fig = go.Figure()
        bloch_fig.add_trace(go.Scatter3d(
            x=bloch_points[:,0], y=bloch_points[:,1], z=bloch_points[:,2],
            mode="markers+lines", marker=dict(size=5, color="blue"), name="Quantum States"
        ))
        bloch_fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title="Bloch Sphere Path"
        )
        st.plotly_chart(bloch_fig, use_container_width=True)
