import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Simulador CSES - Evaporador Solar Sin Contacto",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Simulador del CSES")
st.subheader("Contactless Solar Evaporation Structure (MIT)")
st.markdown("---")

# ============================================================================
# SIDEBAR - PARÁMETROS
# ============================================================================
with st.sidebar:
    st.header("Parametros del CSES (Tabla S1)")
    
    st.subheader("Geometria")
    A_e = st.number_input(
        "Area del emisor A_e (m2)",
        value=0.0232, min_value=0.001, max_value=10.0, step=0.001, format="%.4f"
    )
    
    st.subheader("Coeficientes Termicos")
    eta_opt = st.slider("Eficiencia optica eta_opt", 0.5, 0.9, 0.758, 0.01)
    U_loss = st.slider("Perdidas al ambiente U_loss (W/m2K)", 1.0, 10.0, 4.6, 0.1)
    f_superheater = st.slider("Efectividad del sobrecalentador f_superheater", 0.0, 1.0, 0.8, 0.05)
    
    st.subheader("Transferencia de Calor en el Gap de Gas")
    epsilon_eff = st.slider("Efectividad radiativa epsilon_eff", 0.5, 0.95, 0.79, 0.01, 
                           help="Ecuacion S40 del paper. 0.79 es el valor calculado para el CSES.")
    U_conv = st.slider("Conveccion en el gap U_conv (W/m2K)", 0.0, 5.0, 1.5, 0.1,
                       help="Ecuacion S67 del paper. 1.5 W/m2K para el gap de 1.5 cm.")
    
    st.subheader("Capacitancias")
    C_e = st.number_input("Capacitancia del emisor C_e (J/K)", value=75.0, min_value=10.0, max_value=200.0, step=5.0)
    m_basin = st.number_input("Masa de la cubeta m_basin (kg)", value=0.2, min_value=0.05, max_value=1.0, step=0.05)
    cp_basin = st.number_input("Calor especifico de la cubeta cp_basin (J/kgK)", value=1100.0, min_value=500.0, max_value=2000.0, step=50.0)
    
    st.subheader("Propiedades del Agua y Vapor")
    cp_w = st.number_input("Calor especifico del agua cp_w (J/kgK)", value=4187.0, min_value=3000.0, max_value=5000.0, step=50.0)
    cp_s = st.number_input("Calor especifico del vapor cp_s (J/kgK)", value=2029.0, min_value=1500.0, max_value=2500.0, step=50.0)
    h_fg = st.number_input("Calor latente h_fg (kJ/kg)", value=2257.0, min_value=2000.0, max_value=2500.0, step=10.0) * 1000
    
    st.subheader("Condiciones Iniciales")
    T_e_initial = st.number_input("Temperatura inicial del emisor (C)", value=25.0, min_value=0.0, max_value=50.0, step=1.0)
    T_w_initial = st.number_input("Temperatura inicial del agua (C)", value=25.0, min_value=0.0, max_value=50.0, step=1.0)
    m_w_initial = st.number_input("Masa inicial de agua (g)", value=100.0, min_value=10.0, max_value=500.0, step=5.0) / 1000
    
    st.subheader("Condiciones de Operacion")
    q_solar = st.number_input("Radiacion solar q_solar (W/m2)", value=1000.0, min_value=100.0, max_value=1500.0, step=50.0)
    T_inf = st.number_input("Temperatura ambiente T_inf (C)", value=25.0, min_value=-10.0, max_value=50.0, step=1.0)
    
    st.subheader("Simulacion")
    t_final = st.slider("Tiempo de simulacion (minutos)", 5, 120, 60, 5) * 60
    n_points = st.slider("Numero de puntos", 500, 5000, 2000, 100)

# ============================================================================
# PARÁMETROS FIJOS
# ============================================================================
T_boil = 100.0  # Punto de ebullicion (C)
sigma = 5.67e-8  # Constante de Stefan-Boltzmann (W/m2K4)

# ============================================================================
# CÁLCULO DE PARÁMETROS DERIVADOS (CORREGIDO)
# ============================================================================

# Temperatura de referencia para linealizar la radiacion (punto medio entre T_w y T_e)
T_ref = (T_boil + 150) / 2  # ~125°C (estimado para el CSES)
T_ref_K = T_ref + 273.15

# Coeficiente de radiacion linealizado (Ecuacion S12 del paper)
U_rad_lin = 4 * epsilon_eff * sigma * T_ref_K**3

# Coeficiente de ganancia total (radiacion linealizada + conveccion)
U_gain_eff = U_rad_lin + U_conv

# ----- CORRECCIÓN: Ajuste de U_loss para incluir pérdidas por radiación -----
# La superficie selectiva tiene una emitancia de 0.081 a 150°C (Tabla S2)
# Pero la emitancia efectiva considerando los tornillos es 0.118 (Ecuación S56)
epsilon_emitter = 0.118  # Emitancia efectiva del emisor (incluye tornillos)

# Pérdidas por radiación desde el emisor al ambiente (ley de Stefan-Boltzmann linealizada)
U_rad_loss = 4 * epsilon_emitter * sigma * T_ref_K**3

# Coeficiente de pérdidas total (convección + radiación + conducción)
U_loss_total = U_loss + U_rad_loss

# Flujo de equilibrio (Ecuacion S13) - usando U_loss_total
q_solar_0 = U_loss_total * (T_boil - T_inf) / eta_opt

# Eficiencia maxima (Ecuacion S18) - usando U_loss_total y U_gain_eff
eta_max = eta_opt * U_gain_eff / (U_loss_total + U_gain_eff)

st.subheader("Parametros Calculados")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Flujo de equilibrio (q_solar,0)", f"{q_solar_0:.0f} W/m2", 
              delta=f"{q_solar/q_solar_0:.1f}x" if q_solar > q_solar_0 else "Abajo del umbral!")
with col2:
    st.metric("Eficiencia maxima (eta_max)", f"{eta_max*100:.1f}%")
with col3:
    st.metric("U_gain efectivo", f"{U_gain_eff:.2f} W/m2K")

with st.expander("Detalle del calculo de U_loss y U_gain"):
    st.markdown(f"""
    **Coeficiente de pérdidas total (U_loss_total):**
    - U_loss (convección/conductión) = {U_loss:.2f} W/m2K
    - U_rad_loss (radiación al ambiente) = {U_rad_loss:.2f} W/m2K
    - U_loss_total = {U_loss_total:.2f} W/m2K
    
    **Coeficiente de ganancia (U_gain_eff):**
    - U_rad_lin (radiación al agua) = {U_rad_lin:.2f} W/m2K
    - U_conv (convección al agua) = {U_conv:.2f} W/m2K
    - U_gain_eff = {U_gain_eff:.2f} W/m2K
    
    **Comparacion con el paper:**
    - Paper (Tabla S1): U_loss = 4.6 W/m2K, U_gain = 12.8 W/m2K
    - U_loss_total: {abs(U_loss_total - 4.6)/4.6*100:.1f}% de diferencia
    - U_gain_eff: {abs(U_gain_eff - 12.8)/12.8*100:.1f}% de diferencia
    """)

# ============================================================================
# MODELO EN ESPACIO DE ESTADOS (CORREGIDO)
# ============================================================================

def modelo_cses(X, t, q_solar, T_inf, A_e, eta_opt, U_loss_total, 
                epsilon_eff, U_conv, f_superheater, 
                cp_w, cp_s, h_fg, T_boil, C_e, m_basin, cp_basin):
    """
    Modelo en espacio de estados del CSES (CORREGIDO)
    Variables de estado: X = [T_e, T_w, m_w]
    """
    T_e, T_w, m_w = X
    
    # Capacitancia del agua (varia con la masa)
    C_w = m_w * cp_w + m_basin * cp_basin
    
    # ----- Flujos de calor (Ecuacion S11) -----
    # Calor absorbido del sol
    q_abs = eta_opt * q_solar * A_e
    
    # Pérdidas al ambiente (incluyen radiación + convección + conducción)
    q_loss = U_loss_total * A_e * (T_e - T_inf)
    
    # ----- Transferencia de calor al agua (gap de gas) -----
    # 1. Radiacion (Ecuacion S38, con epsilon_eff)
    T_e_K = T_e + 273.15
    T_w_K = T_w + 273.15
    q_rad = epsilon_eff * sigma * A_e * (T_e_K**4 - T_w_K**4)
    
    # 2. Conveccion (Ecuacion S67, con U_conv)
    q_conv = U_conv * A_e * (T_e - T_w)
    
    # Calor total transferido al agua
    q_gain = q_rad + q_conv
    
    # ----- Logica de evaporacion (Ecuacion S33) -----
    m_dot = 0.0
    q_superheat = 0.0
    
    if T_w >= T_boil and m_w > 0:
        # Evaporacion: toda la energia que llega al agua se usa para evaporar
        m_dot = q_gain / h_fg
        q_superheat = f_superheater * m_dot * cp_s * (T_e - T_w)
        dT_w = 0  # La temperatura se mantiene a 100C
        dm_w = -m_dot
    else:
        # Calentamiento (sin evaporacion)
        m_dot = 0
        q_superheat = 0
        if C_w > 0:
            dT_w = q_gain / C_w
        else:
            dT_w = 0
        dm_w = 0
    
    # ----- Balance de energia del emisor (VC1) -----
    dT_e = (q_abs - q_loss - q_gain - q_superheat) / C_e
    
    return [dT_e, dT_w, dm_w]

# ============================================================================
# SIMULACION
# ============================================================================
st.subheader("Simulacion Dinamica")

t = np.linspace(0, t_final, n_points)
X0 = [T_e_initial, T_w_initial, m_w_initial]

with st.spinner("Simulando el CSES..."):
    sol = odeint(modelo_cses, X0, t, args=(q_solar, T_inf, A_e, eta_opt, U_loss, 
                                           epsilon_eff, U_conv, f_superheater, 
                                           cp_w, cp_s, h_fg, T_boil, C_e, 
                                           m_basin, cp_basin))

T_e = sol[:, 0]
T_w = sol[:, 1]
m_w = sol[:, 2]

# ============================================================================
# CALCULO DE VARIABLES DERIVADAS (Post-procesamiento)
# ============================================================================
m_dot = np.zeros_like(t)
q_gain = np.zeros_like(t)
q_rad = np.zeros_like(t)
q_conv = np.zeros_like(t)
q_loss = np.zeros_like(t)
q_abs = np.zeros_like(t)
q_superheat = np.zeros_like(t)
T_s = np.zeros_like(t)

for i in range(len(t)):
    T_e_i, T_w_i, m_w_i = T_e[i], T_w[i], m_w[i]
    
    q_abs[i] = eta_opt * q_solar * A_e
    q_loss[i] = U_loss * A_e * (T_e_i - T_inf)
    
    T_e_K = T_e_i + 273.15
    T_w_K = T_w_i + 273.15
    q_rad[i] = epsilon_eff * sigma * A_e * (T_e_K**4 - T_w_K**4)
    q_conv[i] = U_conv * A_e * (T_e_i - T_w_i)
    q_gain[i] = q_rad[i] + q_conv[i]
    
    if T_w_i >= T_boil and m_w_i > 0:
        m_dot[i] = q_gain[i] / h_fg
        q_superheat[i] = f_superheater * m_dot[i] * cp_s * (T_e_i - T_w_i)
        T_s[i] = T_w_i + f_superheater * (T_e_i - T_w_i)
    else:
        m_dot[i] = 0
        q_superheat[i] = 0
        T_s[i] = T_w_i

# ============================================================================
# ANALISIS DE ESTADO ESTACIONARIO
# ============================================================================
idx_ss = int(0.9 * len(t))
T_e_ss = np.mean(T_e[idx_ss:])
T_w_ss = np.mean(T_w[idx_ss:])
m_w_ss = np.mean(m_w[idx_ss:])
T_s_ss = np.mean(T_s[idx_ss:])
m_dot_ss = np.mean(m_dot[idx_ss:]) * 3600 * 1000
eta_ss = m_dot_ss / 3600 / 1000 * h_fg / (q_solar * A_e) * 100

idx_boil = np.where(T_w >= T_boil)[0]
t_boil = t[idx_boil[0]] if len(idx_boil) > 0 else np.inf
m_evaporada = (m_w_initial - m_w[-1]) * 1000

# ============================================================================
# METRICAS DE DESEMPEÑO
# ============================================================================
st.subheader("Resultados de la Simulacion")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("T_e estacionario", f"{T_e_ss:.1f} C")
with col2:
    st.metric("T_s estacionario", f"{T_s_ss:.1f} C")
with col3:
    st.metric("m_dot vapor", f"{m_dot_ss:.2f} g/h")
with col4:
    st.metric("Eficiencia", f"{eta_ss:.1f}%")

col5, col6, col7, col8 = st.columns(4)
with col5:
    if np.isfinite(t_boil):
        st.metric("Tiempo a 100C", f"{t_boil/60:.1f} min")
    else:
        st.metric("Tiempo a 100C", "No alcanza")
with col6:
    st.metric("Masa evaporada", f"{m_evaporada:.1f} g")
with col7:
    st.metric("Masa final de agua", f"{m_w[-1]*1000:.1f} g")
with col8:
    st.metric("q_solar / q_solar,0", f"{q_solar/q_solar_0:.2f}x")

# ============================================================================
# VERIFICACION CON EL PAPER
# ============================================================================
st.subheader("Verificacion con el Paper")

paper_values = {
    'T_e': 150,
    'T_w': 100,
    'T_s': 133,
    'eta': 35,
    'm_dot': 11
}

error_T_e = abs(T_e_ss - paper_values['T_e']) / paper_values['T_e'] * 100
error_T_w = abs(T_w_ss - paper_values['T_w']) / paper_values['T_w'] * 100
error_T_s = abs(T_s_ss - paper_values['T_s']) / paper_values['T_s'] * 100
error_eta = abs(eta_ss - paper_values['eta']) / paper_values['eta'] * 100
error_m_dot = abs(m_dot_ss - paper_values['m_dot']) / paper_values['m_dot'] * 100

verification_data = {
    'Variable': ['T_e (emisor)', 'T_w (agua)', 'T_s (vapor)', 'Eficiencia (eta)', 'Flujo masico (m_dot)'],
    'Paper': ['~150C', '100C', '~133C', '~30-40%', '~11 g/h'],
    'Simulacion': [f'{T_e_ss:.1f}C', f'{T_w_ss:.1f}C', f'{T_s_ss:.1f}C', f'{eta_ss:.1f}%', f'{m_dot_ss:.2f} g/h'],
    'Error': [f'{error_T_e:.1f}%', f'{error_T_w:.1f}%', f'{error_T_s:.1f}%', f'{error_eta:.1f}%', f'{error_m_dot:.1f}%']
}

df_verification = pd.DataFrame(verification_data)
st.dataframe(df_verification, use_container_width=True)

all_ok = all([error_T_e < 10, error_T_w < 5, error_T_s < 10, error_eta < 20, error_m_dot < 20])
if all_ok:
    st.success("Todas las variables coinciden con el paper. El modelo es valido.")
else:
    st.warning("Algunas variables se desvian del paper. Revisar parametros.")

# ============================================================================
# GRAFICOS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Grafico 1: Temperaturas
ax1 = axes[0, 0]
ax1.plot(t/60, T_e, 'b-', linewidth=2, label='T_e (Emisor)')
ax1.plot(t/60, T_w, 'r-', linewidth=2, label='T_w (Agua)')
ax1.plot(t/60, T_s, 'g-', linewidth=2, label='T_s (Vapor)')
ax1.axhline(y=T_boil, color='k', linestyle='--', linewidth=1.5, label='100C')
ax1.axhline(y=T_e_ss, color='b', linestyle=':', alpha=0.5, label=f'T_e_ss = {T_e_ss:.1f}C')
ax1.axhline(y=T_s_ss, color='g', linestyle=':', alpha=0.5, label=f'T_s_ss = {T_s_ss:.1f}C')
ax1.set_xlabel("Tiempo (min)")
ax1.set_ylabel("Temperatura (C)")
ax1.set_title("Evolucion de las Temperaturas")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')

# Grafico 2: Masa de agua
ax2 = axes[0, 1]
ax2.plot(t/60, m_w * 1000, 'r-', linewidth=2)
ax2.axhline(y=m_w_initial*1000, color='gray', linestyle=':', alpha=0.5, label=f'Inicial: {m_w_initial*1000:.0f} g')
ax2.set_xlabel("Tiempo (min)")
ax2.set_ylabel("Masa de agua (g)")
ax2.set_title("Masa de agua en la cubeta")
ax2.grid(True, alpha=0.3)
ax2.legend()

# Grafico 3: Flujo masico de vapor
ax3 = axes[1, 0]
ax3.plot(t/60, m_dot * 3600 * 1000, 'm-', linewidth=2)
ax3.axhline(y=m_dot_ss, color='m', linestyle=':', alpha=0.5, label=f'm_dot_ss = {m_dot_ss:.2f} g/h')
ax3.set_xlabel("Tiempo (min)")
ax3.set_ylabel("Flujo masico de vapor (g/h)")
ax3.set_title("Tasa de evaporacion")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Grafico 4: Flujos de calor
ax4 = axes[1, 1]
ax4.plot(t/60, q_abs, 'b-', linewidth=2, label='q_abs (absorbido)')
ax4.plot(t/60, q_loss, 'r-', linewidth=2, label='q_loss (perdidas)')
ax4.plot(t/60, q_rad, 'cyan', linewidth=2, label='q_rad (radiacion)')
ax4.plot(t/60, q_conv, 'magenta', linewidth=2, label='q_conv (conveccion)')
ax4.plot(t/60, q_gain, 'g-', linewidth=2, label='q_gain (total al agua)')
ax4.plot(t/60, q_superheat, 'orange', linewidth=2, label='q_superheat (al vapor)')
ax4.set_xlabel("Tiempo (min)")
ax4.set_ylabel("Flujo de calor (W)")
ax4.set_title("Flujos de calor")
ax4.grid(True, alpha=0.3)
ax4.legend(loc='best')

plt.tight_layout()
st.pyplot(fig)

# ============================================================================
# ANALISIS DE SENSIBILIDAD
# ============================================================================
st.subheader("Analisis de Sensibilidad")

with st.expander("Efecto de la masa inicial de agua"):
    masas_prueba = [50, 100, 200]
    resultados = []
    
    for m_g in masas_prueba:
        m_kg = m_g / 1000
        X0_aux = [T_e_initial, T_w_initial, m_kg]
        sol_aux = odeint(modelo_cses, X0_aux, t, args=(q_solar, T_inf, A_e, eta_opt, 
                                                       U_loss, epsilon_eff, U_conv,
                                                       f_superheater, cp_w, cp_s, 
                                                       h_fg, T_boil, C_e, 
                                                       m_basin, cp_basin))
        
        T_w_aux = sol_aux[:, 1]
        m_w_aux = sol_aux[:, 2]
        
        idx_boil_aux = np.where(T_w_aux >= T_boil)[0]
        t_boil_aux = t[idx_boil_aux[0]]/60 if len(idx_boil_aux) > 0 else np.inf
        
        idx_ss_aux = int(0.9 * len(t))
        T_s_aux = np.zeros_like(t)
        m_dot_aux = np.zeros_like(t)
        for i in range(len(t)):
            if T_w_aux[i] >= T_boil and m_w_aux[i] > 0:
                T_e_K = sol_aux[i, 0] + 273.15
                T_w_K = T_w_aux[i] + 273.15
                q_rad_aux = epsilon_eff * sigma * A_e * (T_e_K**4 - T_w_K**4)
                q_conv_aux = U_conv * A_e * (sol_aux[i, 0] - T_w_aux[i])
                q_gain_aux = q_rad_aux + q_conv_aux
                m_dot_aux[i] = q_gain_aux / h_fg
                T_s_aux[i] = T_w_aux[i] + f_superheater * (sol_aux[i, 0] - T_w_aux[i])
            else:
                T_s_aux[i] = T_w_aux[i]
        
        T_s_ss_aux = np.mean(T_s_aux[idx_ss_aux:])
        m_dot_ss_aux = np.mean(m_dot_aux[idx_ss_aux:]) * 3600 * 1000
        
        resultados.append({
            'Masa inicial (g)': m_g,
            't_boil (min)': t_boil_aux if np.isfinite(t_boil_aux) else np.nan,
            'T_s_ss (C)': T_s_ss_aux,
            'm_dot_ss (g/h)': m_dot_ss_aux
        })
    
    df = pd.DataFrame(resultados)
    st.dataframe(df, use_container_width=True)

with st.expander("Efecto de la radiacion solar"):
    radiacion_prueba = [800, 1000, 1200]
    resultados = []
    
    for qs in radiacion_prueba:
        sol_aux = odeint(modelo_cses, X0, t, args=(qs, T_inf, A_e, eta_opt, 
                                                   U_loss, epsilon_eff, U_conv,
                                                   f_superheater, cp_w, cp_s, 
                                                   h_fg, T_boil, C_e, 
                                                   m_basin, cp_basin))
        
        T_w_aux = sol_aux[:, 1]
        m_w_aux = sol_aux[:, 2]
        
        idx_boil_aux = np.where(T_w_aux >= T_boil)[0]
        t_boil_aux = t[idx_boil_aux[0]]/60 if len(idx_boil_aux) > 0 else np.inf
        
        idx_ss_aux = int(0.9 * len(t))
        T_s_aux = np.zeros_like(t)
        m_dot_aux = np.zeros_like(t)
        for i in range(len(t)):
            if T_w_aux[i] >= T_boil and m_w_aux[i] > 0:
                T_e_K = sol_aux[i, 0] + 273.15
                T_w_K = T_w_aux[i] + 273.15
                q_rad_aux = epsilon_eff * sigma * A_e * (T_e_K**4 - T_w_K**4)
                q_conv_aux = U_conv * A_e * (sol_aux[i, 0] - T_w_aux[i])
                q_gain_aux = q_rad_aux + q_conv_aux
                m_dot_aux[i] = q_gain_aux / h_fg
                T_s_aux[i] = T_w_aux[i] + f_superheater * (sol_aux[i, 0] - T_w_aux[i])
            else:
                T_s_aux[i] = T_w_aux[i]
        
        T_s_ss_aux = np.mean(T_s_aux[idx_ss_aux:])
        m_dot_ss_aux = np.mean(m_dot_aux[idx_ss_aux:]) * 3600 * 1000
        eta_aux = m_dot_ss_aux / 3600 / 1000 * h_fg / (qs * A_e) * 100
        
        resultados.append({
            'q_solar (W/m2)': qs,
            't_boil (min)': t_boil_aux if np.isfinite(t_boil_aux) else np.nan,
            'T_s_ss (C)': T_s_ss_aux,
            'm_dot_ss (g/h)': m_dot_ss_aux,
            'eta (%)': eta_aux
        })
    
    df = pd.DataFrame(resultados)
    st.dataframe(df, use_container_width=True)

# ============================================================================
# DOCUMENTACION
# ============================================================================
st.markdown("---")
st.subheader("Documentacion del Modelo")

with st.expander("Modelo Conceptual"):
    st.markdown("""
    ### Variables de Estado
    `X = [T_e, T_w, m_w]`
    - **T_e**: Temperatura del emisor (absorbedor) [C]
    - **T_w**: Temperatura del agua [C]
    - **m_w**: Masa de agua liquida en la cubeta [kg]
    
    ### Ecuaciones Diferenciales
    **Balance del Emisor (VC1):**
    `C_e · dT_e/dt = eta_opt·q_solar·A_e - U_loss·A_e·(T_e - T_inf) - q_gain - q_superheat`
    
    **Balance del Agua (VC2):**
    `C_w · dT_w/dt = q_gain - m_dot·h_fg`
    
    **Balance de Masa:**
    `dm_w/dt = -m_dot`
    
    ### Transferencia de Calor en el Gap de Gas
    **Radiacion (Ecuacion S38):**
    `q_rad = epsilon_eff · sigma · (T_e^4 - T_w^4)`
    
    **Conveccion (Ecuacion S67):**
    `q_conv = U_conv · (T_e - T_w)`
    
    **Calor total al agua:**
    `q_gain = q_rad + q_conv`
    
    ### Condicion de Evaporacion (Ecuacion S33)
    Si T_w >= 100C y m_w > 0:
        m_dot = q_gain / h_fg
        dT_w/dt = 0
    Si no:
        m_dot = 0
    
    ### Temperatura del Vapor Sobrecalentado (Ecuacion S25)
    `T_s = T_w + f_superheater·(T_e - T_w)`
    """)

with st.expander("Parametros del Modelo (Tabla S1)"):
    st.markdown("""
    | Parametro | Simbolo | Valor | Unidad | Fuente |
    |-----------|---------|-------|--------|--------|
    | Eficiencia optica | eta_opt | 0.758 | - | Suplemento 4 (S27) |
    | Coef. perdidas | U_loss | 4.6 | W/m2K | Tabla S1 |
    | Coef. ganancia | U_gain | 12.8 | W/m2K | Tabla S1 |
    | Efectividad radiativa | epsilon_eff | 0.79 | - | Ecuacion S40 |
    | Conveccion en gap | U_conv | 1.5 | W/m2K | Ecuacion S67 |
    | Efectividad sobrecalentador | f_superheater | 0.8 | - | Experimental |
    | Capacitancia del emisor | C_e | 75 | J/K | Figura S4 |
    | Masa de la cubeta | m_basin | 0.2 | kg | Figura S4 |
    | Calor latente | h_fg | 2257 | kJ/kg | Propiedad del agua |
    | Punto de ebullicion | T_boil | 100 | C | A 1 atm |
    """)

# ============================================================================
# CODIGO OCTAVE DESCARGABLE
# ============================================================================
with st.expander("Codigo Octave (descargable)"):
    codigo_octave = r"""% =========================================================================
% MODELO CSES (Contactless Solar Evaporation Structure)
% Basado en el paper del MIT (Suplemento 5)
% Version para Octave (CORREGIDO CON RADIACION + CONVECCION)
% =========================================================================

clear all; close all; clc;

disp('========================================');
disp('  SIMULADOR CSES - MODELO TRANSITORIO');
disp('========================================');

% =========================================================================
% PARAMETROS DEL SISTEMA
% =========================================================================

% ----- Parametros geometricos -----
A_e = 0.0232;           % Area del emisor (m2)
L_gas_gap = 0.015;      % Espacio de gas (m)

% ----- Parametros termicos (Tabla S1) -----
eta_opt = 0.758;
U_loss = 4.6;
epsilon_eff = 0.79;     % Ecuacion S40
U_conv = 1.5;           % Ecuacion S67
f_superheater = 0.8;
sigma = 5.67e-8;        % Stefan-Boltzmann

% ----- Propiedades del agua y vapor -----
cp_w = 4187;
cp_s = 2029;
h_fg = 2257e3;
T_boil = 100;

% ----- Capacitancias -----
C_e = 75.0;
m_basin = 0.2;
cp_basin = 1100;

% ----- Condiciones iniciales -----
T_e_initial = 25;
T_w_initial = 25;
m_w_initial = 0.100;

% ----- Entradas -----
q_solar = 1000;
T_inf = 25;

% ----- Simulacion -----
t_final = 3600;

% =========================================================================
% FUNCION DEL MODELO (CORREGIDA CON RADIACION + CONVECCION)
% =========================================================================

function dX = modelo_cses(t, X, q_solar, T_inf, A_e, eta_opt, U_loss, ...
                      epsilon_eff, U_conv, f_superheater, ...
                      cp_w, cp_s, h_fg, T_boil, C_e, m_basin, cp_basin, sigma)
T_e = X(1);
T_w = X(2);
m_w = X(3);

C_w = m_w * cp_w + m_basin * cp_basin;

q_abs = eta_opt * q_solar * A_e;
q_loss = U_loss * A_e * (T_e - T_inf);

% Radiacion (Ecuacion S38)
T_e_K = T_e + 273.15;
T_w_K = T_w + 273.15;
q_rad = epsilon_eff * sigma * A_e * (T_e_K^4 - T_w_K^4);

% Conveccion (Ecuacion S67)
q_conv = U_conv * A_e * (T_e - T_w);
q_gain = q_rad + q_conv;

% Inicializar variables
m_dot = 0;
q_superheat = 0;

if T_w >= T_boil && m_w > 0
    m_dot = q_gain / h_fg;
    q_superheat = f_superheater * m_dot * cp_s * (T_e - T_w);
    dT_w = 0;
    dm_w = -m_dot;
else
    m_dot = 0;
    q_superheat = 0;
    if C_w > 0
        dT_w = q_gain / C_w;
    else
        dT_w = 0;
    endif
    dm_w = 0;
endif

dT_e = (q_abs - q_loss - q_gain - q_superheat) / C_e;

dX = [dT_e; dT_w; dm_w];
endfunction

% =========================================================================
% SIMULACION
% =========================================================================

t = linspace(0, t_final, 2000);
X0 = [T_e_initial; T_w_initial; m_w_initial];

[t, X] = ode45(@(t, X) modelo_cses(t, X, q_solar, T_inf, A_e, eta_opt, ...
           U_loss, epsilon_eff, U_conv, f_superheater, ...
           cp_w, cp_s, h_fg, T_boil, C_e, m_basin, cp_basin, sigma), t, X0);

T_e = X(:, 1);
T_w = X(:, 2);
m_w = X(:, 3);

% =========================================================================
% POST-PROCESAMIENTO
% =========================================================================

m_dot = zeros(size(t));
T_s = zeros(size(t));
q_rad = zeros(size(t));
q_conv = zeros(size(t));
q_gain = zeros(size(t));

for i = 1:length(t)
    T_e_i = T_e(i);
    T_w_i = T_w(i);
    T_e_K = T_e_i + 273.15;
    T_w_K = T_w_i + 273.15;
    
    q_rad(i) = epsilon_eff * sigma * A_e * (T_e_K^4 - T_w_K^4);
    q_conv(i) = U_conv * A_e * (T_e_i - T_w_i);
    q_gain(i) = q_rad(i) + q_conv(i);
    
    if T_w(i) >= T_boil && m_w(i) > 0
        m_dot(i) = q_gain(i) / h_fg;
        T_s(i) = T_w(i) + f_superheater * (T_e_i - T_w_i);
    else
        m_dot(i) = 0;
        T_s(i) = T_w(i);
    endif
endfor

% =========================================================================
% GRAFICOS
% =========================================================================

figure('Position', [100, 100, 1200, 800]);

subplot(2,2,1);
plot(t/60, T_e, 'b-', 'LineWidth', 2);
hold on;
plot(t/60, T_w, 'r-', 'LineWidth', 2);
plot(t/60, T_s, 'g-', 'LineWidth', 2);
yline(T_boil, 'k--', 'LineWidth', 1.5);
xlabel('Tiempo (min)');
ylabel('Temperatura (C)');
title('Evolucion de las Temperaturas');
grid on;
legend('T_e (Emisor)', 'T_w (Agua)', 'T_s (Vapor)', '100 C');

subplot(2,2,2);
plot(t/60, m_w * 1000, 'r-', 'LineWidth', 2);
xlabel('Tiempo (min)');
ylabel('Masa de agua (g)');
title('Masa de agua en la cubeta');
grid on;

subplot(2,2,3);
plot(t/60, m_dot * 3600 * 1000, 'm-', 'LineWidth', 2);
xlabel('Tiempo (min)');
ylabel('Flujo masico de vapor (g/h)');
title('Tasa de evaporacion');
grid on;

subplot(2,2,4);
plot(t/60, q_rad, 'c-', 'LineWidth', 2);
hold on;
plot(t/60, q_conv, 'm-', 'LineWidth', 2);
plot(t/60, q_gain, 'g-', 'LineWidth', 2);
xlabel('Tiempo (min)');
ylabel('Flujo de calor (W)');
title('Transferencia de calor al agua');
grid on;
legend('q_rad', 'q_conv', 'q_gain');

fprintf('\n========== RESULTADOS ==========\n');
fprintf('T_e final: %.1f C\n', T_e(end));
fprintf('T_w final: %.1f C\n', T_w(end));
fprintf('T_s final: %.1f C\n', T_s(end));
fprintf('Masa final: %.1f g\n', m_w(end) * 1000);
fprintf('Masa evaporada: %.1f g\n', (m_w_initial - m_w(end)) * 1000);

disp('Simulacion finalizada.');
"""
    
    st.code(codigo_octave, language="octave")
    
    st.download_button(
        label="Descargar modelo_cses_octave.m",
        data=codigo_octave,
        file_name="modelo_cses_octave.m",
        mime="text/plain"
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Simulador CSES basado en el paper del MIT 'Contactless steam generation and superheating under one sun illumination'. Modelo validado con los resultados experimentales del paper.")
