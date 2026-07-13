
### Temperatura del Vapor Sobrecalentado (Ecuación S25)
`T_s = T_w + f_superheater·(T_e - T_w)`
""")

with st.expander("Parámetros del Modelo (Tabla S1)"):
st.markdown("""
| Parámetro | Símbolo | Valor | Unidad | Fuente |
|-----------|---------|-------|--------|--------|
| Eficiencia óptica | η_opt | 0.758 | - | Suplemento 4 (S27) |
| Coef. pérdidas | U_loss | 4.6 | W/m²K | Tabla S1 |
| Efectividad radiativa | ε_eff | 0.79 | - | Ecuación S40 |
| Convección en gap | U_conv | 1.5 | W/m²K | Ecuación S67 |
| Efectividad sobrecalentador | f_superheater | 0.8 | - | Experimental |
| Capacitancia del emisor | C_e | 75 | J/K | Figura S4 |
| Masa de la cubeta | m_basin | 0.2 | kg | Figura S4 |
| Calor latente | h_fg | 2257 | kJ/kg | Propiedad del agua |
| Punto de ebullición | T_boil | 100 | °C | A 1 atm |
""")

# ============================================================================
# CÓDIGO OCTAVE DESCARGABLE
# ============================================================================
with st.expander("📄 Código Octave (descargable)"):
codigo_octave = """% =========================================================================
% MODELO CSES (Contactless Solar Evaporation Structure)
% Basado en el paper del MIT (Suplemento 5)
% Versión para Octave (CORREGIDO CON RADIACION + CONVECCION)
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

t = linspace(0, t_final, 2000)';
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

fprintf('\\n========== RESULTADOS ==========\\n');
fprintf('T_e final: %.1f C\\n', T_e(end));
fprintf('T_w final: %.1f C\\n', T_w(end));
fprintf('T_s final: %.1f C\\n', T_s(end));
fprintf('Masa final: %.1f g\\n', m_w(end) * 1000);
fprintf('Masa evaporada: %.1f g\\n', (m_w_initial - m_w(end)) * 1000);

disp('Simulacion finalizada.');
"""

st.code(codigo_octave, language="octave")

st.download_button(
    label="📥 Descargar modelo_cses_octave.m",
    data=codigo_octave,
    file_name="modelo_cses_octave.m",
    mime="text/plain"
)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
"""
<div style="text-align: center; font-size: 0.8rem; color: #888;">
    Simulador CSES basado en el paper del MIT 
    <a href="https://doi.org/10.1038/s41560-018-0173-2" target="_blank">
        "Contactless steam generation and superheating under one sun illumination"
    </a>
    <br>
    Modelo validado con los resultados experimentales del paper.
</div>
""",
unsafe_allow_html=True
)
