import uuid
import re
import csv
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Analizador de Créditos", layout="wide")

# =========================
# Config / Constantes
# =========================
REQ_BASE1 = ["CEDULA", "CUPO", "FECHA NACIMIENTO", "SEGMENTO"]
COL_DISTINCT = "Distinct ID"

# =========================
# Utilidades
# =========================
def generar_uuid_v5(cedula: str):
    """Hash v5 usando la lógica: {cedula} + 'CED' + {primeros 3 dígitos}"""
    cedula = str(cedula or "").strip()
    if len(cedula) < 3:
        return None
    name = f"{cedula}CED{cedula[:3]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def parse_money_to_float(x):
    """
    Convierte valores tipo:
    7505.93 | 7,505.93 | 7.505,93 | $7 505,93 | 7 505,93 | 7'505.93 -> float
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()

    # limpia símbolos de moneda/espacios raros
    s = (s.replace("$", "")
           .replace("€", "")
           .replace(" ", "")
           .replace("\xa0", "")  # NBSP
           .replace("’", "")
           .replace("'", ""))

    # número simple con punto decimal
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            return float(s)
        except:
            return np.nan

    # sólo comas (LATAM)
    if "," in s and "." not in s:
        s = s.replace(".", "")      # miles
        s = s.replace(",", ".")     # decimal
        try:
            return float(s)
        except:
            return np.nan

    # punto y coma a la vez: decidir por el último separador
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_com > last_dot:
            # decimal con coma
            s = s.replace(".", "")   # miles
            s = s.replace(",", ".")  # decimal
        else:
            # decimal con punto
            s = s.replace(",", "")   # miles
        try:
            return float(s)
        except:
            return np.nan

    # sólo puntos (pueden ser miles): unir si hay muchos
    parts = s.split(".")
    if len(parts) > 2:
        s = "".join(parts)
    try:
        return float(s)
    except:
        return np.nan

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
    """
    Detecta separador automáticamente y lee CSV con fallback de encoding.
    """
    fileobj.seek(0)
    head = fileobj.read(4096).decode("utf-8", errors="ignore")
    # Sniffer
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=",;|\t")
        sep = dialect.delimiter
    except Exception:
        sep = ";" if head.count(";") > head.count(",") else ","
    # volver a inicio y leer
    fileobj.seek(0)
    try:
        df = pd.read_csv(fileobj, dtype=str, sep=sep)
    except UnicodeDecodeError:
        fileobj.seek(0)
        df = pd.read_csv(fileobj, dtype=str, sep=sep, encoding="latin-1")
    return _clean_cols(df), sep

def _find_desembolso_col(cols):
    """
    Busca columna de desembolso con patrones comunes.
    """
    patterns = [
        r"monto[_\s]*a[_\s]*recibir",
        r"desembols",
        r"monto[_\s]*desembols",
        r"sum[_\s]*of[_\s]*monto",
    ]
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        for pat in patterns:
            if re.search(pat, lc):
                return c
    return None

# =========================
# UI
# =========================
st.title("Analizador de Créditos 🚀")
st.caption("Sube la Base 1 (aprobados) y la Base 2 (desembolsados). Se hace hash, cruce y se muestran KPIs, gráficos y tabla.")

col_u1, col_u2 = st.columns(2)
with col_u1:
    base1_file = st.file_uploader("Sube Base 1 (Aprobados) .xlsx", type=["xlsx"])
with col_u2:
    base2_file = st.file_uploader("Sube Base 2 (Desembolsados) .csv / .xlsx", type=["csv", "xlsx", "xls"])

if base1_file is None or base2_file is None:
    st.info("💡 Sube ambos archivos para procesar.")
    st.stop()

# =========================
# Cargar y validar Base 1
# =========================
try:
    base1 = pd.read_excel(base1_file, dtype=str)
    base1 = _clean_cols(base1)
except Exception as e:
    st.error(f"Error leyendo Base 1: {e}")
    st.stop()

faltantes = [c for c in REQ_BASE1 if c not in base1.columns]
if faltantes:
    st.error(f"En Base 1 faltan columnas requeridas: {faltantes}")
    st.stop()

# Hash CEDULA -> Distinct ID
base1[COL_DISTINCT] = base1["CEDULA"].apply(generar_uuid_v5)
# Limpiar CUPO a número
base1["Monto_Ofertado"] = base1["CUPO"].apply(parse_money_to_float)
# Edad
base1["Edad"] = calcular_edad(base1["FECHA NACIMIENTO"])

# =========================
# Cargar y preparar Base 2 (CSV o Excel) - robusto
# =========================
name2 = base2_file.name.lower()
try:
    if name2.endswith(".csv"):
        base2, detected_sep = _guess_csv_df(base2_file)
        selected_sheet = None
    else:
        xls = pd.ExcelFile(base2_file)
        sheets = xls.sheet_names
        selected_sheet = sheets[0]
        if len(sheets) > 1:
            selected_sheet = st.selectbox("Selecciona la hoja de la Base 2 (Excel):", sheets, index=0)
        base2 = pd.read_excel(xls, sheet_name=selected_sheet, dtype=str)
        base2 = _clean_cols(base2)
        detected_sep = None
except Exception as e:
    st.error(f"Error leyendo Base 2: {e}")
    st.stop()

# Normalizamos nombres clave
if COL_DISTINCT not in base2.columns:
    candidatas = [c for c in base2.columns if c.strip().lower().replace(" ", "") in ("distinctid", "distinct_id")]
    if candidatas:
        base2.rename(columns={candidatas[0]: COL_DISTINCT}, inplace=True)

if COL_DISTINCT not in base2.columns:
    st.error(f"En Base 2 no se encontró la columna '{COL_DISTINCT}'. Revisa el nombre exacto.")
    st.stop()

col_desembolso = _find_desembolso_col(base2.columns)
if col_desembolso is None:
    st.error("En Base 2 no encontré la columna de desembolso (busco 'monto_a_recibir', 'desembols*', etc.). Renombra o verifica el archivo.")
    st.stop()

# Tipos y parseo de montos
if "Time" in base2.columns:
    base2["Time"] = to_datetime_safe(base2.get("Time"))

# Copia raw para debug
base2["_raw_desemb"] = base2[col_desembolso]
base2["Monto_Desembolsado"] = base2[col_desembolso].apply(parse_money_to_float)

# Fallback si todo quedó NaN
if base2["Monto_Desembolsado"].isna().all():
    base2["Monto_Desembolsado"] = pd.to_numeric(
        base2[col_desembolso]
        .astype(str)
        .str.replace(r"[\$\s]", "", regex=True)
        .str.replace(".", "", regex=False)  # miles
        .str.replace(",", ".", regex=False),  # decimal
        errors="coerce"
    )

# =========================
# Normalizar IDs y debug de cruce
# =========================
# Normalizar Distinct ID en ambas bases (evita espacios, mayúsculas/minúsculas)
base1[COL_DISTINCT] = base1[COL_DISTINCT].astype(str).str.strip().str.lower()
base2[COL_DISTINCT] = base2[COL_DISTINCT].astype(str).str.strip().str.lower()

with st.expander("🔍 Debug del cruce (Distinct ID)", expanded=True):
    total_b1 = base1[COL_DISTINCT].nunique()
    total_b2 = base2[COL_DISTINCT].nunique()
    inter = len(set(base1[COL_DISTINCT]).intersection(set(base2[COL_DISTINCT])))

    st.write("**Distinct ID únicos (Base 1):**", total_b1)
    st.write("**Distinct ID únicos (Base 2):**", total_b2)
    st.write("**Intersección (IDs que matchean):**", inter)

    if inter == 0:
        st.warning("No hay intersección de IDs. Revisa la lógica de hash o el normalizado de IDs.")
        st.write("**Ejemplos Base 1 (primeros 5):**", base1[COL_DISTINCT].head(5).tolist())
        st.write("**Ejemplos Base 2 (primeros 5):**", base2[COL_DISTINCT].head(5).tolist())

# =========================
# Merge y métricas
# =========================
cols_b2 = [COL_DISTINCT, "Monto_Desembolsado"]
if "Time" in base2.columns:
    cols_b2.append("Time")

# df_all: mantiene todos los aprobados (para tasa de conversión)
df_all = base1.merge(base2[cols_b2], on=COL_DISTINCT, how="left")

df_all["Diferencia"] = df_all["Monto_Ofertado"] - df_all["Monto_Desembolsado"]
df_all["% Utilización"] = np.where(
    df_all["Monto_Ofertado"] > 0,
    (df_all["Monto_Desembolsado"] / df_all["Monto_Ofertado"]) * 100,
    np.nan
)

# df_acc: solo los que desembolsaron (IDs de Base 2)
df_acc = df_all[df_all["Monto_Desembolsado"].notna() & (df_all["Monto_Desembolsado"] > 0)].copy()

# =========================
# Panel de métricas (sobre df_all)
# =========================
total_ofertado = df_all["Monto_Ofertado"].sum(skipna=True)
total_desemb = df_all["Monto_Desembolsado"].sum(skipna=True)
prom_util = df_all["% Utilización"].mean(skipna=True)
tasa_conv = (df_all["Monto_Desembolsado"].notna() & (df_all["Monto_Desembolsado"] > 0)).mean() * 100

st.subheader("Resumen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Promedio de Utilización", f"{prom_util:,.1f}%")
m2.metric("Total Ofertado", f"${total_ofertado:,.0f}")
m3.metric("Total Desembolsado", f"${total_desemb:,.0f}")
m4.metric("Tasa de Conversión", f"{tasa_conv:,.1f}%")

# =========================
# Filtros (aplican a ambos dataframes)
# =========================
with st.expander("🎛️ Filtros"):
    segs = sorted([s for s in df_all["SEGMENTO"].dropna().unique().tolist() if str(s).strip() != ""]) if "SEGMENTO" in df_all.columns else []
    seg_sel = st.multiselect("Segmento", segs, default=segs) if segs else []
    edad_min = int(np.nanmin(df_all["Edad"])) if np.isfinite(np.nanmin(df_all["Edad"])) else 18
    edad_max = int(np.nanmax(df_all["Edad"])) if np.isfinite(np.nanmax(df_all["Edad"])) else 80
    r_edad = st.slider("Rango de edad", min_value=0, max_value=max(18, edad_max),
                       value=(max(18, edad_min), max(18, edad_max)))

mask_all = (
    (df_all["SEGMENTO"].isin(seg_sel) if seg_sel else True) &
    df_all["Edad"].between(r_edad[0], r_edad[1], inclusive="both")
)
mask_acc = (
    (df_acc["SEGMENTO"].isin(seg_sel) if seg_sel else True) &
    df_acc["Edad"].between(r_edad[0], r_edad[1], inclusive="both")
)

df_all_f = df_all.loc[mask_all].copy()
df_acc_f = df_acc.loc[mask_acc].copy()

# =========================
# Gráficos (sobre df_all filtrado)
# =========================
st.subheader("Gráficos")

col_g1, col_g2 = st.columns(2)
with col_g1:
    aux = df_all_f["% Utilización"].dropna()
    if len(aux):
        fig = px.histogram(aux, x="% Utilización", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos para el histograma de % Utilización.")

with col_g2:
    if "SEGMENTO" in df_all_f.columns and len(df_all_f):
        seg_mean = df_all_f.groupby("SEGMENTO")["% Utilización"].mean().reset_index()
        fig = px.bar(seg_mean.sort_values("% Utilización"), x="% Utilización", y="SEGMENTO", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

bins = [18, 25, 35, 45, 55, 65, 120]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
df_all_f["Grupo_Edad"] = pd.cut(df_all_f["Edad"], bins=bins, labels=labels, right=True, include_lowest=True)
edad_mean = df_all_f.groupby("Grupo_Edad")["% Utilización"].mean().reset_index()
fig = px.bar(edad_mean, x="Grupo_Edad", y="% Utilización")
st.plotly_chart(fig, use_container_width=True)

scatter = df_all_f.dropna(subset=["Monto_Ofertado", "Monto_Desembolsado"])
if len(scatter):
    fig = px.scatter(
        scatter,
        x="Monto_Ofertado",
        y="Monto_Desembolsado",
        color="SEGMENTO" if "SEGMENTO" in scatter.columns else None,
        hover_data=[COL_DISTINCT, "Edad"]
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Tabla detalle (SOLO IDs que desembolsaron)
# =========================
st.subheader("Detalle — Solo clientes con desembolso (Base 2)")

cols_show = [COL_DISTINCT, "SEGMENTO", "Edad", "Monto_Ofertado", "Monto_Desembolsado", "Diferencia", "% Utilización"]
if "Time" in df_acc_f.columns:
    cols_show.append("Time")
disp = df_acc_f[cols_show].copy()

# Formato amigable
for c in ["Monto_Ofertado", "Monto_Desembolsado", "Diferencia"]:
    disp[c] = disp[c].apply(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
disp["% Utilización"] = disp["% Utilización"].apply(lambda v: "" if pd.isna(v) else f"{v:,.1f}%")

st.dataframe(disp, use_container_width=True)
