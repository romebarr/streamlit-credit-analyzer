import re

def parse_money_to_float(x):
    """
    Convierte valores tipo:
    7505.93 | 7,505.93 | 7.505,93 | $7 505,93 | 7 505,93 | 7'505.93
    -> float
    """
    if pd.isna(x): 
        return np.nan
    s = str(x).strip()

    # Limpia símbolos de moneda/espacios raros
    s = (s.replace("$", "")
           .replace("€", "")
           .replace(" ", "")
           .replace("\xa0", "")  # NBSP
           .replace("’", "")
           .replace("'", ""))

    # Si ya es un número simple con punto decimal (caso 7505.93)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            return float(s)
        except:
            return np.nan

    # Si solo tiene comas -> formato LATAM (7.505,93 o 7505,93)
    if "," in s and "." not in s:
        s = s.replace(".", "")      # miles (por si viniera 7.505,93)
        s = s.replace(",", ".")     # decimal
        try:
            return float(s)
        except:
            return np.nan

    # Si tiene punto y coma a la vez -> decide por el último separador
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_com > last_dot:
            # decimal con coma (7,505.93 rara vez; aquí sería 7.505,93 realmente)
            s = s.replace(".", "")  # miles
            s = s.replace(",", ".") # decimal
        else:
            # decimal con punto (7,505.93)
            s = s.replace(",", "")  # miles
        try:
            return float(s)
        except:
            return np.nan

    # Si solo tiene puntos (puede ser miles o decimal)
    # Caso más común: 7505.93 -> ya habría pasado por el primer if,
    # aquí limpiamos miles del tipo 7.505 (sin decimales)
    parts = s.split(".")
    if len(parts) > 2:
        # muchos puntos: asumimos puntos como miles, sin decimales
        s = "".join(parts)
    try:
        return float(s)
    except:
        return np.nan
