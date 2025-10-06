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
def generar_uuid_v5_default(cedula: str):
    """Lógica original: {cedula} + 'CED' + {primeros 3 dígitos}, namespace DNS."""
    cedula = str(cedula or "").strip()
    if len(cedula) < 3:
        return ""
    name = f"{cedula}CED{cedula[:3]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))  # mantiene guiones, lowercase

def parse_money_to_float(x):
    """
    Convierte valores tipo:
    7505.93 | 7,505.93 | 7.505,93 | $7 505,93 | 7 505,93 | 7'505.93 -> float
    """
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
        try:
            return float(s)
        except:
            return np.nan

    if "," in s and "." not in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
        try:
            return float(s)
        except:
            return np.nan

    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_com > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        try:
            return float(s)
        except:
            return np.nan

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
    # Solo limpia encabezados (NO toca valores)
    df.columns = [c.strip() for c in df.columns]
    return df

def _guess_csv_df(fileobj):
    """
    Detecta separador automáticamente y lee CSV con fallback de encoding.
    NO toca valores de columnas.
    """
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
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        for pat in patterns:
            if re.search(pat, lc):
                return c
    return None

# ======= HASH CANDIDATES (Auto-detección SIN normalizar IDs) =======
NAMESPACES = {
    "DNS": uuid.NAMESPACE_DNS,
    "URL": uuid.NAMESPACE_URL,
    "OID": uuid.NAMESPACE_OID,
    "X500": uuid.NAMESPACE_X500,
}
def name_formula(ced: str, variant: str) -> str:
    if variant == "ced+CED+first3":
        return f"{ced}CED{ced[:3]}"
    if variant == "ced+CED+last3":
        return f"{ced}CED{ced[-3:]}"
    if variant == "ced:CED:first3":
        return f"{ced}:CED:{ced[:3]}"
    if variant == "ced-ced-first3":
        return f"{ced}-CED-{ced[:3]}"
    if variant == "cedcedfirst3_lowerced":
        return f"{ced}ced{ced[:3]}"
    return f"{ced}CED{ced[:3]}"

FORMULAS = [
    "ced+CED+first3",
    "ced+CED+last3",
    "ced:CED:first3",
    "ced-ced-first3",
    "cedcedfirst3_lowerced",
]

def hash_with(cedula: str, ns_key: str, formula_key: str) -> str:
    ced = str(cedula or "").strip()
    if len(ced) < 3:
        return ""
    nm = name_formula(ced, formula_key)
    return str(uuid.uuid5(NAMESPACES[ns_key], nm))  # mantiene guiones

def autodetect_hash_exact(base1_ced_series: pd.Series, base2_ids_exact_set: set, sample_n: int = 300):
    """
    Prueba combinaciones de namespace x fórmula sobre una muestra de cédulas,
    y compara EXACTAMENTE contra los Distinct ID de Base 2 (sin normalizar).
    """
    sample = base1_ced_series.dropna().astype(str).unique()[:sample_n]
    results = []
    for ns in NAMESPACES.keys():
        for fm in FORMULAS:
            gen = [hash_with(ced, ns, fm) for ced in sample]  # string con guiones
            inter = len(set(gen).intersection(base2_ids_exact_set))
            results.append((ns, fm, inter))
    results.sort(key=lambda x: x[2], reverse=True)
    return results

# =========================
# UI
# =========================
st.title("Analizador de Créditos 🚀")
st.caption("Sube la Base 1 (aprobados) y la Base 2 (desembolsados). Se hace hash (sin alterar el formato del ID), cruce y se muestran KPIs, gráficos y tabla.")

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
    base1 = _clean_cols(base1)  # limpia solo encabezados
except Exception as e:
    st.error(f"Error leyendo Base 1: {e}")
    st.stop()

faltantes = [c for c in REQ_BASE1 if c not in base1.columns]
if faltantes:
    st.error(f"En Base 1 faltan columnas requeridas: {faltantes}")
    st.stop()

# Cálculos base (NO tocamos Distinct ID aún)
base1["Monto_Ofertado"] = base1["CUPO"].apply(parse_money_to_float)
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

# Distinct ID debe existir (NO lo tocamos)
if COL_DISTINCT not in base2.columns:
    candidatas = [c for c in base2.columns if c.strip().lower().replace(" ", "") in ("distinctid", "distinct_id")]
    if candidatas:
        base2.rename(columns={candidatas[0]: COL_DISTINCT}, inplace=True)

if COL_DISTINCT not in base2.columns:
    st.error(f"En Base 2 no se encontró la columna '{COL_DISTINCT}'. Revisa el nombre exacto.")
    st.stop()

# Detectar columna de desembolso
col_desembolso = _find_desembolso_col(base2.columns)
if col_desembolso is None:
    st.error("En Base 2 no encontré la columna de desembolso (busco 'monto_a_recibir', 'desembols*', etc.).")
    st.stop()

# Parseo de Time y monto (NO tocamos valores de Distinct ID)
if "Time" in base2.columns:
    base2["Time"] = to_datetime_safe(base2.get("Time"))

base2["_raw_desemb"] = base2[col_desembolso]
base2["Monto_Desembolsado"] = base2[col_desembolso].apply(parse_money_to_float)
if base2["Monto_Desembolsado"].isna().all():
    base2["Monto_Desembolsado"] = pd.to_numeric(
        base2[col_desembolso]
        .astype(str)
        .str.replace(r"[\$\s]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

# =========================
# Debug Base 2 (parseo)
# =========================
with st.expander("🧪 Debug Base 2 (parseo)", expanded=False):
    st.write("**Columna detectada de desembolso:**", col_desembolso)
    if detected_sep is not None:
        st.write("**Separador CSV detectado:**", detected_sep)
    if selected_sheet is not None:
        st.write("**Hoja seleccionada (Excel):**", selected_sheet)
    st.write("**No nulos (raw):**", int(base2["_raw_desemb"].notna().sum()))
    st.write("**No nulos (parseados):**", int(base2["Monto_Desembolsado"].notna().sum()))
    st.write("**Muestras raw:**")
    st.write(base2[["_raw_desemb"]].head(5))
    st.write("**Suma parseada Monto_Desembolsado:**", float(base2["Monto_Desembolsado"].sum(skipna=True)))

# =========================
# AUTODETECCIÓN de HASH (sin normalizar Distinct ID)
# =========================
b2_ids_exact_set = set(base2[COL_DISTINCT].dropna().astype(str))

tests = autodetect_hash_exact(base1["CEDULA"], b2_ids_exact_set, sample_n=500)

with st.expander("🧬 Auto-detección de hash (namespace + fórmula)", expanded=True):
    if len(tests):
        df_tests = pd.DataFrame(tests, columns=["Namespace", "Fórmula", "Matches (muestra)"])
        st.dataframe(df_tests, use_container_width=True)
    else:
        st.write("Sin tests disponibles.")

best_ns, best_fm, best_inter = tests[0] if tests else ("DNS", "ced+CED+first3", 0)

if best_inter == 0:
    st.error("No hubo matches en la auto-detección de hash. El Distinct ID de la Base 2 podría no provenir de la cédula con UUID v5 (o usa otra fórmula/namespace).")
    # Aun así generamos con la fórmula por defecto para continuar con la interfaz
    base1[COL_DISTINCT] = base1["CEDULA"].apply(generar_uuid_v5_default)
else:
    st.success(f"Mejor combinación detectada: Namespace={best_ns} | Fórmula={best_fm} (matches en muestra: {best_inter})")
    base1[COL_DISTINCT] = base1["CEDULA"].apply(lambda c: hash_with(c, best_ns, best_fm))

# =========================
# Debug del cruce (EXACTO, sin normalizar IDs)
# =========================
with st.expander("🔍 Debug del cruce (Distinct ID exacto)", expanded=True):
    total_b1 = base1[COL_DISTINCT].nunique()
    total_b2 = base2[COL_DISTINCT].nunique()
    inter = len(set(base1[COL_DISTINCT]).intersection(b2_ids_exact_set))

    st.write("**Distinct ID únicos (Base 1):**", total_b1)
    st.write("**Distinct ID únicos (Base 2):**", total_b2)
    st.write("**Intersección (IDs que matchean, exactos):**", inter)

    if inter == 0:
        st.warning("Sigue sin haber intersección exacta. Verifica que Base 2 realmente use UUID v5 derivado de la cédula con la misma fórmula/namespace.")
        st.write("**Ejemplos Base 1 (primeros 5):**", base1[COL_DISTINCT].head(5).tolist())
        st.write("**Ejemplos Base 2 (primeros 5):**", base2[COL_DISTINCT].head(5).tolist())

# =========================
# Merge y métricas (match EXACTO)
# =========================
cols_b2 = [COL_DISTINCT, "Monto_Desembolsado"]
if "Time" in base2.columns:
    cols_b2.append("Time")

df_all = base1.merge(base2[cols_b2], on=COL_DISTINCT, how="left")

df_all["Diferencia"] = df_all["Monto_Ofertado"] - df_all["Monto_Desembolsado"]
df_all["% Utilización"] = np.where(
    df_all["Monto_Ofertado"] > 0,
    (df_all["Monto_Desembolsado"] / df_all["Monto_Ofertado"]) * 100,
    np.nan
)

# Solo los que desembolsaron (IDs Base 2)
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
# Filtros
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
# Tabla detalle — Solo colocados (IDs Base 2, tal cual)
# =========================
st.subheader("Detalle — Solo clientes con desembolso (Base 2)")
cols_show = [COL_DISTINCT, "SEGMENTO", "Edad", "Monto_Ofertado", "Monto_Desembolsado", "Diferencia", "% Utilización"]
if "Time" in df_acc_f.columns:
    cols_show.append("Time")

disp = df_acc_f[cols_show].copy()
for c in ["Monto_Ofertado", "Monto_Desembolsado", "Diferencia"]:
    disp[c] = disp[c].apply(lambda v: "" if pd.isna(v) else f"${v:,.0f}")
disp["% Utilización"] = disp["% Utilización"].apply(lambda v: "" if pd.isna(v) else f"{v:,.1f}%")
st.dataframe(disp, use_container_width=True)
