import uuid
import re
import csv
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Analizador de Créditos", layout="wide")

# =========================
# CONFIGURACIÓN
# =========================
REQ_BASE1 = ["CEDULA", "CUPO", "FECHA NACIMIENTO", "SEGMENTO"]
COL_DISTINCT = "Distinct ID"

# =========================
# FUNCIONES
# =========================
def generar_uuid_v5(cedula: str):
    """Lógica EXACTA: cedula + 'CED' + primeros 3 dígitos → UUIDv5 (DNS)"""
    cedula = str(cedula or "").strip()
    if len(cedula) < 3:
        return ""
    name = f"{cedula}CED{cedula[:3]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def parse_money_to_float(x):
    """Convierte montos con distintos formatos a float"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = (s.replace("$", "")
           .replace("€", "")
           .replace(" ", "")
           .replace("\xa0", "")
           .replace("’", "")
           .replace("'", ""))
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        try: return float(s)
        except: return np.nan
    if "," in s and "." not in s:
        s = s.replace(".", "").replace(",", ".")
        try: return float(s)
        except: return np.nan
    if "." in s and "," in s:
        last_dot, last_com = s.rfind("."), s.rfind(",")
        if last_com > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
        try: return float(s)
        except: return np.nan
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts)
    try: return float(s)
    except: return np.nan

def to_datetime_safe(s):
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if isinstance(dt, pd.Series) and pd.isna(dt).all():
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return dt

def calcular_edad(fecha_nac_series):
    fechas = to_datetime_safe(fecha_nac_series)
    hoy = pd.Timestamp.today().normalize()
    return (hoy - fechas).dt.days // 365

def _clean_cols(df: pd.DataFrame):
    df.columns = [c.strip() for c in df.columns]
    return df

def _guess_csv_df(fileobj):
    """Detecta separador automáticamente"""
    fileobj.seek(0)
    head = fileobj.read(4096).decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;|\t")
        sep = dialect.delimiter
    except Exception:
        sep = ";" if head.count(";") > head.count(",") else ","
    fileobj.seek(0)
    try:
        df = pd.read_csv(fileobj, dtype=str, sep=sep)
    except UnicodeDecodeError:
        fileobj.seek(0)
        df = pd.read_csv(fileobj, dtype=str, sep=sep, encoding="latin-1")
    return _clean_cols(df), sep

def _find_desembolso_col(cols):
    patterns = [
        r"monto[_\s]*a[_\s]*recibir",
        r"desembols",
        r"monto[_\s]*desembols",
        r"sum[_\s]*of[_\s]*monto",
    ]
    for c in cols:
        lc = c.lower()
        if any(re.search(p, lc) for p in patterns):
            return c
    return None

# =========================
# UI
# =========================
st.title("Analizador de Créditos 🚀")
st.caption("Compara el monto ofertado vs. desembolsado usando el Distinct ID generado con UUIDv5 (DNS).")

col1, col2 = st.columns(2)
with col1:
    base1_file = st.file_uploader("📘 Base 1 (Aprobados, .xlsx)", type=["xlsx"])
with col2:
    base2_file = st.file_uploader("📗 Base 2 (Desembolsados, .csv o .xlsx)", type=["csv", "xlsx", "xls"])

if base1_file is None or base2_file is None:
    st.info("💡 Sube ambos archivos para procesar.")
    st.stop()

# =========================
# BASE 1
# =========================
try:
    base1 = pd.read_excel(base1_file, dtype=str)
    base1 = _clean_cols(base1)
except Exception as e:
    st.error(f"Error leyendo Base 1: {e}")
    st.stop()

faltantes = [c for c in REQ_BASE1 if c not in base1.columns]
if faltantes:
    st.error(f"Faltan columnas requeridas en Base 1: {faltantes}")
    st.stop()

# Cálculos Base 1
base1["Monto_Ofertado"] = base1["CUPO"].apply(parse_money_to_float)
base1["Edad"] = calcular_edad(base1["FECHA NACIMIENTO"])
base1[COL_DISTINCT] = base1["CEDULA"].apply(generar_uuid_v5)

# =========================
# BASE 2
# =========================
name2 = base2_file.name.lower()
try:
    if name2.endswith(".csv"):
        base2, _ = _guess_csv_df(base2_file)
    else:
        xls = pd.ExcelFile(base2_file)
        base2 = pd.read_excel(xls, sheet_name=xls.sheet_names[0], dtype=str)
        base2 = _clean_cols(base2)
except Exception as e:
    st.error(f"Error leyendo Base 2: {e}")
    st.stop()

if COL_DISTINCT not in base2.columns:
    candidatas = [c for c in base2.columns if c.strip().lower().replace(" ", "") in ("distinctid", "distinct_id")]
    if candidatas:
        base2.rename(columns={candidatas[0]: COL_DISTINCT}, inplace=True)

if COL_DISTINCT not in base2.columns:
    st.error(f"No se encontró la columna '{COL_DISTINCT}' en Base 2.")
    st.stop()

col_desembolso = _find_desembolso_col(base2.columns)
if not col_desembolso:
    st.error("No se encontró la columna de monto desembolsado (busco 'monto_a_recibir', 'desembols*', etc.)")
    st.stop()

base2["Monto_Desembolsado"] = base2[col_desembolso].apply(parse_money_to_float)
if "Time" in base2.columns:
    base2["Time"] = to_datetime_safe(base2["Time"])

# =========================
# CRUCE EXACTO
# =========================
df = base1.merge(
    base2[[COL_DISTINCT, "Monto_Desembolsado"] + (["Time"] if "Time" in base2.columns else [])],
    on=COL_DISTINCT, how="left"
)

# Métricas de match
match_count = df["Monto_Desembolsado"].notna().sum()
total_ids_b1 = base1[COL_DISTINCT].nunique()
total_ids_b2 = base2[COL_DISTINCT].nunique()

st.success(f"✅ Distinct ID únicos Base 1: {total_ids_b1:,} | Base 2: {total_ids_b2:,} | Matches encontrados: {match_count:,}")

# =========================
# CÁLCULOS
# =========================
df["Diferencia"] = df["Monto_Ofertado"] - df["Monto_Desembolsado"]
df["% Utilización"] = np.where(
    df["Monto_Ofertado"] > 0,
    (df["Monto_Desembolsado"] / df["Monto_Ofertado"]) * 100,
    np.nan
)

df_des = df[df["Monto_Desembolsado"].notna() & (df["Monto_Desembolsado"] > 0)].copy()

# =========================
# MÉTRICAS GENERALES
# =========================
total_ofertado = df["Monto_Ofertado"].sum(skipna=True)
total_desemb = df["Monto_Desembolsado"].sum(skipna=True)
prom_util = df["% Utilización"].mean(skipna=True)
tasa_conv = (df["Monto_Desembolsado"].notna() & (df["Monto_Desembolsado"] > 0)).mean() * 100

st.subheader("📊 Resumen General")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Promedio Utilización", f"{prom_util:,.1f}%")
m2.metric("Total Ofertado", f"${total_ofertado:,.0f}")
m3.metric("Total Desembolsado", f"${total_desemb:,.0f}")
m4.metric("Tasa Conversión", f"{tasa_conv:,.1f}%")

# =========================
# FILTROS
# =========================
with st.expander("🎛️ Filtros"):
    segs = sorted(df["SEGMENTO"].dropna().unique()) if "SEGMENTO" in df.columns else []
    seg_sel = st.multiselect("Segmento", segs, default=segs)
    edad_min = int(np.nanmin(df["Edad"])) if np.isfinite(np.nanmin(df["Edad"])) else 18
    edad_max = int(np.nanmax(df["Edad"])) if np.isfinite(np.nanmax(df["Edad"])) else 80
    r_edad = st.slider("Rango de edad", 0, max(18, edad_max), (edad_min, edad_max))

mask = (
    (df["SEGMENTO"].isin(seg_sel) if seg_sel else True)
    & df["Edad"].between(r_edad[0], r_edad[1])
)
df_f = df[mask].copy()
df_des_f = df_des[mask].copy()

# =========================
# GRÁFICOS
# =========================
st.subheader("📈 Visualizaciones")

col_g1, col_g2 = st.columns(2)
with col_g1:
    if len(df_f):
        fig = px.histogram(df_f, x="% Utilización", nbins=30, title="Distribución de % Utilización")
        st.plotly_chart(fig, use_container_width=True)
with col_g2:
    if "SEGMENTO" in df_f.columns:
        seg_mean = df_f.groupby("SEGMENTO")["% Utilización"].mean().reset_index()
        fig = px.bar(seg_mean, x="% Utilización", y="SEGMENTO", orientation="h", title="Utilización Promedio por Segmento")
        st.plotly_chart(fig, use_container_width=True)

bins = [18, 25, 35, 45, 55, 65, 120]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
df_f["Grupo_Edad"] = pd.cut(df_f["Edad"], bins=bins, labels=labels, include_lowest=True)
edad_mean = df_f.groupby("Grupo_Edad")["% Utilización"].mean().reset_index()
fig = px.bar(edad_mean, x="Grupo_Edad", y="% Utilización", title="Utilización Promedio por Grupo de Edad")
st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLA FINAL
# =========================
st.subheader("📋 Detalle — Solo clientes con desembolso (Base 2)")

# Incluimos CÉDULA para poder validar manualmente
cols_show = [
    "CEDULA",
    COL_DISTINCT,
    "SEGMENTO",
    "Edad",
    "Monto_Ofertado",
    "Monto_Desembolsado",
    "Diferencia",
    "% Utilización"
]
if "Time" in df_des_f.columns:
    cols_show.append("Time")

# Verificamos que las columnas existan
cols_show = [c for c in cols_show if c in df_des_f.columns]

disp = df_des_f[cols_show].copy()

# Formateo visual
for c in ["Monto_Ofertado", "Monto_Desembolsado", "Diferencia"]:
    if c in disp.columns:
        disp[c] = disp[c].apply(lambda v: "" if pd.isna(v) else f"${v:,.0f}")

if "% Utilización" in disp.columns:
    disp["% Utilización"] = disp["% Utilización"].apply(lambda v: "" if pd.isna(v) else f"{v:,.1f}%")

st.dataframe(disp, use_container_width=True)
