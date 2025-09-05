from fastapi import HTTPException, APIRouter, Query
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional, Dict, Any
import os, math, asyncio, uuid
import httpx
from dotenv import load_dotenv
import re
import unicodedata
from datetime import datetime, timezone  # timestamps de auditoría Shipday

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  Falta GOOGLE_MAPS_API_KEY en el entorno (.env)")

PLACES_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
PLACES_DETAILS_URL      = "https://maps.googleapis.com/maps/api/place/details/json"
GEOCODE_URL             = "https://maps.googleapis.com/maps/api/geocode/json"
DISTANCE_MATRIX_URL     = "https://maps.googleapis.com/maps/api/distancematrix/json"

# País por defecto → Colombia
PLACES_COUNTRIES = os.getenv("PLACES_COUNTRIES_CO", "co")

# =========================
#  CONFIGURACIÓN DE PRECIOS
# =========================
# Norma (todas las distancias en KM):
#  0–BASE_DISTANCE_KM  -> BASE_PRICE_COP
#  BASE_DISTANCE_KM–TIER1_MAX_KM: TIER1_RATE_PER_KM_COP por km adicional
#  >TIER1_MAX_KM: TIER2_RATE_PER_KM_COP por km adicional extra
#
# 'DISTANCE_BILLING_MODE':
#   - 'ceil'   : cobra km iniciados (recomendado)
#   - 'floor'  : km completos
#   - 'round'  : km redondeados

BASE_DISTANCE_KM            = float(os.getenv("BASE_DISTANCE_KM", "3"))
BASE_PRICE_COP              = int(os.getenv("BASE_PRICE_COP", "5500"))
TIER1_MAX_KM                = float(os.getenv("TIER1_MAX_KM", "8"))
TIER1_RATE_PER_KM_COP       = int(os.getenv("TIER1_RATE_PER_KM_COP", "800"))
TIER2_RATE_PER_KM_COP       = int(os.getenv("TIER2_RATE_PER_KM_COP", "700"))
DISTANCE_BILLING_MODE       = os.getenv("DISTANCE_BILLING_MODE", "ceil")  # 'ceil' | 'floor' | 'round'
DISTANCE_REPORT_DECIMALS    = int(os.getenv("DISTANCE_REPORT_DECIMALS", "2"))

# ======= CONFIG SHIPDAY =======
SHIPDAY_API_KEY_COLOMBIA = os.getenv("SHIPDAY_API_KEY_COLOMBIA", "").strip()
SHIPDAY_DISTANCE_UNITS = (os.getenv("SHIPDAY_DISTANCE_UNITS", "km") or "km").strip().lower()  # 'km' o 'mi'

def _shipday_headers() -> Dict[str, str]:
    if not SHIPDAY_API_KEY_COLOMBIA:
        raise HTTPException(status_code=500, detail="No se configuró SHIPDAY_API_KEY_COLOMBIA")
    return {
        "Authorization": f"Basic {SHIPDAY_API_KEY_COLOMBIA}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _units_is_km() -> bool:
    return SHIPDAY_DISTANCE_UNITS in {"km", "kilometer", "kilometre", "kilometros", "kilómetros"}

def _to_km_from_shipday(v: float) -> float:
    # Shipday devuelve 'distance' en millas o km dependiendo de la cuenta.
    if _units_is_km():
        return float(v)
    # millas -> km
    return float(v) * 1.609344

# ======= NUEVO: disponibilidad Shipday (metadatos) =======
SHIPDAY_AVAILABILITY_URL = "https://api.shipday.com/on-demand/availability"
SHIPDAY_PREFERRED_PROVIDER = (os.getenv("SHIPDAY_PREFERRED_PROVIDER", "uber") or "uber").strip().lower()

def _as_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        m = re.search(r"\d+", val)
        return int(m.group(0)) if m else None
    return None

def _provider_name(item: Dict[str, Any]) -> str:
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    candidates: List[str] = []
    keys = (
        "provider", "providerName", "name", "service", "serviceName",
        "deliveryPartner", "serviceProviderName", "channel", "source",
        "thirdPartyDeliveryService", "deliveryServiceName"
    )
    for k in keys:
        v = item.get(k)
        if isinstance(v, dict):
            for kk in ("name", "provider", "displayName", "title"):
                vv = v.get(kk)
                if isinstance(vv, str):
                    candidates.append(vv)
        elif isinstance(v, str):
            candidates.append(v)
    return " ".join(sorted({_norm(x) for x in candidates if x}))

# Router con prefijo para Colombia
router = APIRouter(prefix="/co")

SEDES: List[Dict[str, Any]] = [
    {
        "site_id": 11,
        "site_name": "LAURELES",
        "site_address": "Transversal 39 #74B-14, Medellín, Antioquia, Colombia",
        "pe_site_id": 4,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Medellín",
                "unit": None,
                "state": "Antioquia",
                "street": "Transversal 39 #74B-14",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 2,
        "site_name": "FLORA",
        "site_address": "Carrera 32 #8-42, ACOPI Yumbo, Yumbo, Valle del Cauca, Colombia",
        "pe_site_id": 1,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Yumbo",
                "unit": None,
                "state": "Valle del Cauca",
                "street": "Carrera 32 #8-42",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 4,
        "site_name": "JAMUNDI",
        "site_address": "Carrera 22 # 5A Sur-60, Jamundí, Valle del Cauca, Colombia",
        "pe_site_id": 6,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Jamundí",
                "unit": None,
                "state": "Valle del Cauca",
                "street": "Carrera 22 #5A Sur-60",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 29,
        "site_name": "CITY U",
        "site_address": "Calle 19 #2A-10, Bogotá, Colombia",
        "pe_site_id": 11,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Bogotá",
                "unit": None,
                "state": "Bogotá D.C.",
                "street": "Calle 19 #2A-10",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 8,
        "site_name": "SUBA",
        "site_address": "Carrera 92 #147B-17, Bogotá, Colombia",
        "pe_site_id": 9,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Bogotá",
                "unit": None,
                "state": "Bogotá D.C.",
                "street": "Carrera 92 #147B-17",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 9,
        "site_name": "MONTES",
        "site_address": "Calle 8 Sur #32A-08, Bogotá, Colombia",
        "pe_site_id": 5,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Bogotá",
                "unit": None,
                "state": "Bogotá D.C.",
                "street": "Calle 8 Sur #32A-08",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 1,
        "site_name": "BRETAÑA",
        "site_address": "Calle 10 #21-42, Bretaña, Cali, Valle del Cauca, Colombia",
        "pe_site_id": 2,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Cali",
                "unit": None,
                "state": "Valle del Cauca",
                "street": "Calle 10 #21-42",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 10,
        "site_name": "KENNEDY",
        "site_address": "Carrera 78B #38 Sur-79, Bogotá, Colombia",
        "pe_site_id": 8,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Bogotá",
                "unit": None,
                "state": "Bogotá D.C.",
                "street": "Carrera 78B #38 Sur-79",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 30,
        "site_name": "USAQUEN",
        "site_address": "Carrera 19 #150-69, Bogotá, Colombia",
        "pe_site_id": 13,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Bogotá",
                "unit": None,
                "state": "Bogotá D.C.",
                "street": "Carrera 19 #150-69",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 3,
        "site_name": "BOCHALEMA",
        "site_address": "Calle 48 # 106-83 Arbolatta mall LOCAL 10",
        "pe_site_id": 7,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Cali",
                "unit": None,
                "state": "Valle del Cauca",
                "street": "Carrera 85 #37-10",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 7,
        "site_name": "MODELIA",
        "site_address": "Carrera 75 #25C-45, Bogotá, Colombia",
        "pe_site_id": 3,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
                "address": {
                "zip": None,
                "city": "Bogotá",
                "unit": None,
                "state": "Bogotá D.C.",
                "street": "Carrera 75 #25C-45",
                "country": "CO",
            }
        },
    },
    {
        "site_id": 2,
        "site_name": "FLORA",
        "site_address": "Calle 44 Norte Av 3E-89, La Flora, Cali, Valle del Cauca, Colombia",
        "pe_site_id": 1,
        "time_zone": "America/Bogota",
        "location": None,
        "city": None,
        "pickup": {
            "address": {
                "zip": None,
                "city": "Cali",
                "unit": None,
                "state": "Valle del Cauca",
                "street": "Calle 44 Norte #3E-89",
                "country": "CO",
            }
        },
    },
]

# ─────────── Helpers de texto ───────────

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def _norm_token(s: str) -> str:
    s = _strip_accents(s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ─────────── Canonicalización de ciudades (CO) ───────────
# Ciudad canónica -> sinónimos y localidades/barrios/comunas que deben mapear a esa ciudad
CITY_CHILDREN_CO: Dict[str, Dict[str, set]] = {
    "bogota": {
        "aliases": {"bogota", "bogota dc", "bogota d.c.", "bogota distrito capital", "bogota d c"},
        "children": {
            "fontibon","suba","kennedy","usaquen","engativa","bosa","ciudad bolivar","san cristobal",
            "teusaquillo","chapinero","santa fe","puente aranda","rafael uribe uribe","tunjuelito",
            "antonio nariño","barrios unidos","la candelaria","los martires","sumapaz","usme",
            "modelia","ciudad salitre","niza","cedritos","enelcodito"
        }
    },
    "medellin": {
        "aliases": {"medellin"},
        "children": {
            "el poblado","laureles","laureles-estadio","estadio","belen","robledo","manrique","boston",
            "buenos aires","aranjuez","villa hermosa","castilla","guayabal","san javier","doce de octubre"
        }
    },
    "cali": {
        "aliases": {"cali","santiago de cali"},
        "children": {
            "la flora","versalles","san fernando","san nicolas","puente del comercio","aguablanca",
            "caney","ciudad jardin","napoles","el ingenio","pance","el prado"
        }
    },
    "yumbo": {
        "aliases": {"yumbo"},
        "children": {"acopi","acopi yumbo","la estancia","las américas","bellavista","urbanizacion acopi"}
    },
    "jamundi": {
        "aliases": {"jamundi","jamundí"},
        "children": {"alfaguara","potrerito","robles"}
    },
}

# Índices rápidos para búsqueda inversa
CITY_CANON_SET = set(CITY_CHILDREN_CO.keys())
CHILD_TO_PARENT_CO: Dict[str, str] = {}
for canon, data in CITY_CHILDREN_CO.items():
    for a in data["aliases"]:
        CHILD_TO_PARENT_CO[_norm_token(a)] = canon
    for child in data["children"]:
        CHILD_TO_PARENT_CO[_norm_token(child)] = canon

def normalize_city(txt: Optional[str]) -> str:
    if not txt:
        return ""
    key = _norm_token(txt)
    parent = CHILD_TO_PARENT_CO.get(key)
    if parent:
        return parent
    if key in CITY_CANON_SET:
        return key
    return key  # fallback limpio

# ─────────── Helpers país ───────────

def _components_for(countries: Optional[str] = None) -> Optional[str]:
    """
    Convierte 'co|us' o 'colombia, usa' -> 'country:co|country:us'
    """
    countries = countries or PLACES_COUNTRIES
    raw = [t.strip().lower() for t in re.split(r"[,\| ]+", countries) if t.strip()]

    alias = {
        "colombia": "co",
        "usa": "us", "u.s.a.": "us", "u.s.": "us", "eeuu": "us",
        "estados-unidos": "us", "estadosunidos": "us",
    }

    norm: List[str] = []
    for t in raw:
        t = alias.get(t, t)
        if re.fullmatch(r"[a-z]{2}", t):  # solo alpha-2
            norm.append(t)

    norm = list(dict.fromkeys(norm))
    if not norm:
        return None
    return "|".join(f"country:{c}" for c in norm)

# ─────────── Modelos ───────────

class DistanceRequest(BaseModel):
    origin: Optional[str] = Field(None, description="Dirección de origen")
    destination: Optional[str] = Field(None, description="Dirección de destino")
    origin_place_id: Optional[str] = Field(None, description="Place ID de origen")
    destination_place_id: Optional[str] = Field(None, description="Place ID de destino")
    session_token: Optional[str] = Field(None, description="Token de sesión de Google Places")

class GeocodedPoint(BaseModel):
    query: str
    formatted_address: str
    lat: float
    lng: float

class DistanceResponse(BaseModel):
    origin: GeocodedPoint
    destination: GeocodedPoint
    distance_km: float                          # SIEMPRE en kilómetros
    distance_miles_driving: Optional[float] = None  # Solo si viene de ruta conducida (no haversine)
    method: str = "google_distance_matrix_driving"

class NearestInfo(BaseModel):
    site: Dict[str, Any]
    distance_km: float                          # Distancia base (haversine) en km para selección
    in_coverage: bool
    driving_distance_km: Optional[float] = None # Ruta conducida si se pudo calcular

class AutocompleteLiteItem(BaseModel):
    description: str
    place_id: str
    types: List[str] = []

class AutocompleteLiteResponse(BaseModel):
    predictions: List[AutocompleteLiteItem]
    session_token: str

class CoverageError(BaseModel):
    code: str
    message_es: str
    message_en: str

class Address(BaseModel):
    unit: Optional[str] = None
    street: str
    city: str
    state: str
    zip: str
    country: str

# ======= Metadatos de disponibilidad Shipday =======
class ShipdayAvailability(BaseModel):
    provider: str
    fee: float
    pickup_duration_minutes: Optional[int] = None
    delivery_duration_minutes: Optional[int] = None
    pickup_time_iso: Optional[str] = None
    delivery_time_iso: Optional[str] = None

# ⬇️ Dropoff ahora incluye raw_address para auditoría
class Dropoff(BaseModel):
    address: Address
    raw_address: Optional[str] = None

class CoverageDetailsResponse(BaseModel):
    place_id: str
    formatted_address: str
    lat: float
    lng: float
    nearest: Optional[NearestInfo] = None
    delivery_cost_cop: Optional[int] = None
    distance_miles: Optional[float] = None          # Distancia de conducción reportada (o fallback) en km
    dropoff: Optional[Dropoff] = None

    # ===== Metadatos/auditoría Shipday (no afectan el precio) =====
    pickup_duration_minutes: Optional[int] = None
    delivery_duration_minutes: Optional[int] = None
    pickup_time_iso: Optional[str] = None
    delivery_time_iso: Optional[str] = None
    shipday_payload: Optional[Dict[str, Any]] = None
    shipday_response: Optional[Any] = None
    shipday_requested_at_iso: Optional[str] = None

    error: Optional[CoverageError] = None

class ShipdayDistanceResponse(BaseModel):
    place_id: str
    formatted_address: str
    lat: float
    lng: float
    nearest: Optional[NearestInfo] = None
    shipday_distance_km: Optional[float] = None          # Normalizado a km SIEMPRE
    shipday_distance_miles_driving: Optional[float] = None  # Derivado de la ruta de Shipday (km->mi)
    error: Optional[CoverageError] = None

# ─────────── Utilidades ───────────

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distancia del círculo máximo (no conducida) en millas.
    (Fórmula corregida)
    """
    R_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_miles * c

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return haversine_miles(lat1, lon1, lat2, lon2) * 1.609344

async def geocode_address_with_components(client: httpx.AsyncClient, address: str, countries: Optional[str] = None) -> Tuple[float, float, str, List[Dict[str, Any]]]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {"address": address, "key": GOOGLE_API_KEY}
    comps = _components_for(countries or "co")
    if comps:
        params["components"] = comps
    resp = await client.get(GEOCODE_URL, params=params, timeout=1000.0)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error geocodificando '{address}': HTTP {resp.status_code}")
    data = resp.json()
    status = data.get("status")
    if status != "OK" or not data.get("results"):
        msg = data.get("error_message") or status or "Sin resultados"
        raise HTTPException(status_code=400, detail=f"No se pudo geocodificar '{address}': {msg}")
    top = data["results"][0]
    loc = top["geometry"]["location"]
    formatted = top.get("formatted_address", address)
    comps_list = top.get("address_components", []) or []
    return float(loc["lat"]), float(loc["lng"]), formatted, comps_list

async def geocode_address(client: httpx.AsyncClient, address: str, countries: Optional[str] = None) -> Tuple[float, float, str]:
    lat, lng, formatted, _ = await geocode_address_with_components(client, address, countries)
    return lat, lng, formatted

async def places_details(client: httpx.AsyncClient, place_id: str, session_token: Optional[str] = None) -> Tuple[float, float, str]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {"place_id": place_id, "key": GOOGLE_API_KEY, "fields": "formatted_address,geometry/location"}
    if session_token:
        params["sessiontoken"] = session_token
    resp = await client.get(PLACES_DETAILS_URL, params=params, timeout=1000.0)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error en Place Details: HTTP {resp.status_code}")
    data = resp.json()
    status = data.get("status")
    if status != "OK" or not data.get("result"):
        msg = data.get("error_message") or status or "Sin resultados"
        raise HTTPException(status_code=400, detail=f"No se pudo resolver place_id '{place_id}': {msg}")
    res = data["result"]
    loc = res["geometry"]["location"]
    formatted = res.get("formatted_address", "")
    return float(loc["lat"]), float(loc["lng"]), formatted or place_id

async def places_details_with_components(client: httpx.AsyncClient, place_id: str, session_token: Optional[str] = None) -> Tuple[float, float, str, List[Dict[str, Any]]]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    # Importante: 'address_component' (singular) como field, pero la respuesta devuelve 'address_components'
    params = {"place_id": place_id, "key": GOOGLE_API_KEY, "fields": "formatted_address,geometry/location,address_component"}
    if session_token:
        params["sessiontoken"] = session_token
    resp = await client.get(PLACES_DETAILS_URL, params=params, timeout=1000.0)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error en Place Details: HTTP {resp.status_code}")
    data = resp.json()
    status = data.get("status")
    if status != "OK" or not data.get("result"):
        msg = data.get("error_message") or status or "Sin resultados"
        raise HTTPException(status_code=400, detail=f"No se pudo resolver place_id '{place_id}': {msg}")
    res = data["result"]
    loc = res["geometry"]["location"]
    formatted = res.get("formatted_address", "")
    comps = res.get("address_components", []) or []
    return float(loc["lat"]), float(loc["lng"]), (formatted or place_id), comps

def _find_component(comps: List[Dict[str, Any]], type_name: str, short: bool = True) -> str:
    for c in comps:
        if type_name in (c.get("types") or []):
            return c.get("short_name") if short and c.get("short_name") else c.get("long_name", "")
    return ""

def extract_city_from_components(comps: List[Dict[str, Any]]) -> str:
    """Intenta extraer la 'ciudad' y normaliza a su ciudad padre si aplica (CO)."""
    locality = _find_component(comps, "locality", short=False)  # 'Bogotá'
    subloc   = _find_component(comps, "sublocality", short=False)  # 'Fontibón'
    aal2     = _find_component(comps, "administrative_area_level_2", short=False)  # 'Bogotá' o 'Valle del Cauca'
    country  = _find_component(comps, "country", short=True)  # 'CO'
    aal1     = _find_component(comps, "administrative_area_level_1", short=False)  # 'Bogotá D.C.' o departamento

    candidate = locality or subloc or aal2 or ""

    if (country or "").upper() == "CO":
        parent = CHILD_TO_PARENT_CO.get(_norm_token(candidate))
        if parent:
            return parent
        # Bogotá a veces viene en AAL1 como 'Bogotá D.C.' → normaliza a 'bogota'
        if _norm_token(aal1) in CHILD_TO_PARENT_CO:
            return CHILD_TO_PARENT_CO[_norm_token(aal1)]

    return normalize_city(candidate)

def _build_address_from_components(comps: List[Dict[str, Any]]) -> Tuple['Address', str]:
    street_number = _find_component(comps, "street_number", short=False)
    route = _find_component(comps, "route", short=False)
    street = f"{street_number} {route}".strip() if (street_number or route) else ""

    city_raw = extract_city_from_components(comps)
    state = _find_component(comps, "administrative_area_level_1", short=False) or ""  # Depto en CO
    zip_code = _find_component(comps, "postal_code", short=False) or ""
    country = _find_component(comps, "country", short=True) or "CO"
    unit = _find_component(comps, "subpremise", short=False) or None

    # Limpieza extra
    street = re.sub(r"\s+", " ", street).strip()
    city = re.sub(r"\s+", " ", city_raw).strip()

    addr = Address(unit=unit, street=street, city=city, state=state, zip=zip_code, country=country)

    # ST_NUM, ST_NAME, CITY, STATE, ZIP, COUNTRY
    st_num = street_number or ""
    st_name = route or ""
    raw_address = f"{st_num}, {st_name}, {city or ''}, {state or ''}, {zip_code or ''}, {country or ''}"

    return addr, raw_address

async def driving_distance_km(client: httpx.AsyncClient, o_lat: float, o_lng: float, d_lat: float, d_lng: float, language: str = "es") -> Tuple[float, int]:
    """
    Distancia de conducción en KM usando Google Distance Matrix.
    Retorna (kilómetros, segundos_estimados).
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {
        "origins": f"{o_lat},{o_lng}",
        "destinations": f"{d_lat},{d_lng}",
        "mode": "driving",
        "units": "metric",  # value = metros; 'units' solo afecta string
        "language": language,
        "key": GOOGLE_API_KEY,
    }
    resp = await client.get(DISTANCE_MATRIX_URL, params=params, timeout=1000.0)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Distance Matrix HTTP {resp.status_code}")
    data = resp.json()
    if data.get("status") != "OK":
        raise HTTPException(status_code=400, detail=f"Distance Matrix: {data.get('status')}")
    el = (data.get("rows") or [{}])[0].get("elements") or [{}]
    el = el[0]
    if el.get("status") != "OK":
        raise HTTPException(status_code=400, detail=f"Distance Matrix element: {el.get('status')}")
    meters  = el["distance"]["value"]
    seconds = el["duration"]["value"]
    km      = meters / 1000.0
    return float(km), int(seconds)

# =============== PRECIO POR TRAMOS =================

def _billable_units(x: float, mode: str) -> int:
    """Devuelve unidades de km a cobrar para un tramo (x >= 0)."""
    x = max(0.0, x)
    if mode == "floor":
        return int(math.floor(x + 1e-9))
    if mode == "round":
        return int(round(x))
    # default 'ceil' (km iniciado)
    return int(math.ceil(x - 1e-9))

def compute_delivery_cost_cop(distance_km: float) -> int:
    """
    Calcula el costo de domicilio en COP según los tramos configurados (todo en KM).
    """
    if distance_km <= 0:
        return BASE_PRICE_COP

    # km adicionales en cada tramo (sin negativos)
    seg1_raw = max(0.0, min(distance_km, TIER1_MAX_KM) - BASE_DISTANCE_KM)
    seg2_raw = max(0.0, distance_km - TIER1_MAX_KM)

    seg1_units = _billable_units(seg1_raw, DISTANCE_BILLING_MODE)
    seg2_units = _billable_units(seg2_raw, DISTANCE_BILLING_MODE)

    total = BASE_PRICE_COP + seg1_units * TIER1_RATE_PER_KM_COP + seg2_units * TIER2_RATE_PER_KM_COP
    return int(total)

# ─────────── Geocodificar SEDES: coords + ciudad ───────────

async def ensure_sites_coords(client: httpx.AsyncClient) -> None:
    for s in SEDES:
        if not s.get("location") or not s.get("city"):
            try:
                lat, lng, _fmt, comps = await geocode_address_with_components(client, s["site_address"], countries="co")
                s["location"] = {"lat": lat, "long": lng}
                raw_city = extract_city_from_components(comps)  # ya devuelve normalizado
                s["city"] = raw_city or s.get("city") or ""
            except Exception:
                # Ignorar sede si falla
                s["location"] = None
                s["city"] = s.get("city") or ""
        # Asegura campo normalizado (útil si ya venía s["city"] prellenado)
        s["city"] = normalize_city(s.get("city"))
        s["city_norm"] = s["city"]

def nearest_among_same_city(lat: float, lng: float, city_norm: str) -> NearestInfo:
    """
    Selecciona SIEMPRE la sede más cercana (Haversine) ENTRE las sedes cuya ciudad normalizada
    coincide con la ciudad del dropoff. Si no hay coincidencias, in_coverage=False.
    Devuelve distancia en KM (no conducida) solo para elegir sede.
    """
    best = None
    best_dist = float("inf")
    for s in SEDES:
        if not s.get("location"):
            continue
        s_city_norm = s.get("city_norm") or normalize_city(s.get("city"))
        if s_city_norm and s_city_norm == city_norm:
            slat = s["location"]["lat"]
            slng = s["location"]["long"]
            d_km = haversine_km(lat, lng, slat, slng)
            if d_km < best_dist:
                best_dist = d_km
                best = s
    if best is None:
        return NearestInfo(site={}, distance_km=float("inf"), in_coverage=False)
    return NearestInfo(site=best, distance_km=round(best_dist, 2), in_coverage=True)

def make_out_of_coverage_error_by_city(city: str) -> CoverageError:
    city_txt = city or "la ciudad indicada"
    return CoverageError(
        code="OUT_OF_COVERAGE",
        message_es=f"No hay cobertura en {city_txt}. Aún no tenemos sedes en esa ciudad.",
        message_en=f"No coverage in {city_txt}. We don't have locations in that city yet.",
    )

# ─────────── HELPERS SHIPDAY: pedido efímero para distancia ───────────

async def _shipday_insert_order_for_distance(
    client: httpx.AsyncClient,
    order_number: str,
    pickup_address_str: str,
    dropoff_address_str: str,
    pickup_name: str = "Pickup",
    dropoff_name: str = "Customer",
    pickup_phone: str = "+573000000000",
    dropoff_phone: str = "+573000000000",
) -> int:
    """
    Inserta un pedido mínimo para que Shipday calcule 'distance'.
    Devuelve orderId.
    """
    url = "https://api.shipday.com/orders"
    headers = _shipday_headers()

    payload = {
        "orderNumber": order_number,
        "customerName": dropoff_name,
        "customerAddress": dropoff_address_str,
        "customerPhoneNumber": dropoff_phone,
        "restaurantName": pickup_name,
        "restaurantAddress": pickup_address_str,
        "restaurantPhoneNumber": pickup_phone,
        # datos opcionales, no disparan ningún carrier
        "schedule": False,
        "paymentMethod": "cash",
        "totalOrderCost": 0,
    }

    r = await client.post(url, headers=headers, json=payload, timeout=30.0)
    if r.status_code >= 300:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise HTTPException(status_code=502, detail=f"Shipday Insert Order fallo: {msg}")
    data = r.json() or {}
    if not data.get("success"):
        raise HTTPException(status_code=502, detail=f"Shipday Insert Order no exitoso: {data}")
    oid = data.get("orderId")
    if not oid:
        raise HTTPException(status_code=502, detail="Shipday Insert Order no retornó orderId")
    return int(oid)

async def _shipday_fetch_distance_by_order_number(
    client: httpx.AsyncClient,
    order_number: str
) -> float:
    """
    Lee /orders/{orderNumber} y toma el 'distance' del primer objeto (unidad de la cuenta).
    """
    url = f"https://api.shipday.com/orders/{order_number}"
    headers = _shipday_headers()
    r = await client.get(url, headers=headers, timeout=30.0)
    if r.status_code >= 300:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise HTTPException(status_code=502, detail=f"Shipday Retrieve Order Details fallo: {msg}")
    arr = r.json() or []
    if not isinstance(arr, list) or not arr:
        raise HTTPException(status_code=502, detail="Shipday no devolvió detalles de orden")
    distance_val = (arr[0] or {}).get("distance")
    if distance_val is None:
        raise HTTPException(status_code=502, detail="Shipday no incluyó campo 'distance'")
    return float(distance_val)

async def _shipday_delete_order(client: httpx.AsyncClient, order_id: int) -> None:
    """
    Borra el pedido efímero para no ensuciar el dashboard.
    """
    url = f"https://api.shipday.com/orders/{order_id}"
    headers = _shipday_headers()
    try:
        await client.delete(url, headers=headers, timeout=30.0)
    except Exception:
        # no bloquear por error de limpieza
        pass

async def _shipday_probe_distance(
    client: httpx.AsyncClient,
    pickup_address_str: str,
    dropoff_address_str: str,
    pickup_name: str = "Pickup"
) -> Tuple[float, float]:
    """
    Crea pedido efímero -> lee 'distance' -> borra. Devuelve (distance_raw, distance_km).
    """
    order_number = f"DIST-{uuid.uuid4().hex[:8]}"
    order_id = await _shipday_insert_order_for_distance(
        client=client,
        order_number=order_number,
        pickup_address_str=pickup_address_str,
        dropoff_address_str=dropoff_address_str,
        pickup_name=pickup_name,
        dropoff_name="Distance Probe",
    )
    try:
        distance_raw = await _shipday_fetch_distance_by_order_number(client, order_number)
    finally:
        await _shipday_delete_order(client, order_id)
    return float(distance_raw), _to_km_from_shipday(distance_raw)

# ======= helper para metadatos de disponibilidad Shipday por direcciones =======
class _SDDeliveryQuote(BaseModel):
    pickupAddress: str
    deliveryAddress: str
    deliveryTime: Optional[str] = None

async def shipday_quote_meta_by_address(
    client: httpx.AsyncClient,
    pickup_address: str,
    delivery_address: str,
    delivery_time_iso: Optional[str] = None,
) -> Tuple[Optional[ShipdayAvailability], Optional[Dict[str, Any]], Optional[Any], str]:
    """
    Devuelve (best_availability, payload_enviado, respuesta_cruda, timestamp_iso_utc).
    No se usa 'fee' para precio (el cálculo sigue con compute_delivery_cost_cop).
    """
    ts_iso = datetime.now(timezone.utc).isoformat()
    if not SHIPDAY_API_KEY_COLOMBIA:
        return None, None, None, ts_iso

    headers = _shipday_headers()
    payload: Dict[str, Any] = {
        "pickupAddress": pickup_address,
        "deliveryAddress": delivery_address,
    }
    if delivery_time_iso:
        payload["deliveryTime"] = delivery_time_iso

    try:
        resp = await client.post(SHIPDAY_AVAILABILITY_URL, headers=headers, json=payload, timeout=1000.0)
    except Exception:
        return None, payload, None, ts_iso

    if resp.status_code != 200:
        raw = None
        try:
            raw = resp.json()
        except Exception:
            raw = {"_non_json_body": resp.text}
        return None, payload, raw, ts_iso

    try:
        data = resp.json()
    except Exception:
        return None, payload, {"_non_json_body": resp.text}, ts_iso

    if not isinstance(data, list):
        return None, payload, data, ts_iso

    preferred = SHIPDAY_PREFERRED_PROVIDER or "uber"
    best: Optional[ShipdayAvailability] = None
    for item in data:
        if not item or item.get("error") is True:
            continue
        fee = item.get("fee")
        if not isinstance(fee, (int, float)):
            continue
        name = _provider_name(item)
        if preferred in name:
            current = ShipdayAvailability(
                provider=preferred,
                fee=float(fee),
                pickup_duration_minutes=_as_int(item.get("pickupDuration")),
                delivery_duration_minutes=_as_int(item.get("deliveryDuration")),
                pickup_time_iso=item.get("pickupTime"),
                delivery_time_iso=item.get("deliveryTime"),
            )
            if best is None or current.fee < best.fee:
                best = current

    return best, payload, data, ts_iso

# ─────────── Endpoints ───────────

@router.get("/places/autocomplete", response_model=AutocompleteLiteResponse)
async def places_autocomplete(
    input: str = Query(..., min_length=1, description="Texto parcial de dirección"),
    session_token: Optional[str] = Query(None, description="Token de sesión para mejor facturación/resultados"),
    language: str = Query("es", description="Idioma de resultados"),
    countries: Optional[str] = Query("co", description="Códigos ISO-3166-1 alpha-2 separados por | o , (p.ej. 'co')"),
    limit: int = Query(5, ge=1, le=10, description="Máximo de sugerencias"),
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")

    stoken = session_token or str(uuid.uuid4())
    comps = _components_for(countries)

    params = {"input": input, "key": GOOGLE_API_KEY, "language": language, "sessiontoken": stoken, "types": "address"}
    if comps:
        params["components"] = comps

    async with httpx.AsyncClient() as client:
        resp = await client.get(PLACES_AUTOCOMPLETE_URL, params=params, timeout=15.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Error en Autocomplete: HTTP {resp.status_code}")
        data = resp.json()
        status = data.get("status")
        if status == "ZERO_RESULTS":
            return AutocompleteLiteResponse(predictions=[], session_token=stoken)
        if status != "OK":
            msg = data.get("error_message") or status or "Error en Autocomplete"
            raise HTTPException(status_code=400, detail=msg)
        raw_preds = data.get("predictions", [])[:limit]
        items: List[AutocompleteLiteItem] = [
            AutocompleteLiteItem(description=p.get("description", ""), place_id=p.get("place_id", ""), types=p.get("types", []) or [])
            for p in raw_preds
        ]
    return AutocompleteLiteResponse(predictions=items, session_token=stoken)

@router.get("/places/coverage-details", response_model=CoverageDetailsResponse)
async def coverage_details(
    place_id: str = Query(..., description="Place ID seleccionado por el usuario"),
    session_token: Optional[str] = Query(None, description="Token de sesión usado en Autocomplete"),
    language: str = Query("es", description="Idioma para Distance Matrix"),
    delivery_time_iso: Optional[str] = Query(None, description="Hora de entrega ISO-8601 (UTC), opcional para Shipday")
):
    """
    Cobertura por CIUDAD (precio por tramos en COP) + metadatos Shipday (opcional).
    - Precio se calcula SIEMPRE con compute_delivery_cost_cop(distance_km).
    - Shipday solo aporta tiempos/fechas y auditoría; su 'fee' NO se usa para cobrar.
    - Todas las distancias de respuesta se reportan en KM; cuando haya millas, son millas conducidas derivadas.
    """
    async with httpx.AsyncClient() as client:
        # 0) Asegurar datos de sedes (coords + city_norm)
        await ensure_sites_coords(client)

        # 1) Resolver coordenadas + components del destino
        dlat, dlng, formatted, comps = await places_details_with_components(client, place_id, session_token)

        # 2) Construir dropoff.address + raw_address
        address, raw_address = _build_address_from_components(comps)
        drop_city_norm = normalize_city(address.city)
        dropoff = Dropoff(address=address, raw_address=raw_address)

        # 3) Buscar sede(s) de la MISMA CIUDAD y ELEGIR LA MÁS CERCANA (ASIGNACIÓN)
        near = nearest_among_same_city(dlat, dlng, drop_city_norm)

        if not near.in_coverage or not near.site:
            return CoverageDetailsResponse(
                place_id=place_id,
                formatted_address=formatted,
                lat=dlat,
                lng=dlng,
                nearest=None,
                delivery_cost_cop=None,
                distance_km=None,
                dropoff=dropoff,
                # metadatos Shipday nulos
                pickup_duration_minutes=None,
                delivery_duration_minutes=None,
                pickup_time_iso=None,
                delivery_time_iso=None,
                shipday_payload=None,
                shipday_response=None,
                shipday_requested_at_iso=None,
                error=make_out_of_coverage_error_by_city(address.city),
            )

        # 4) Dentro de cobertura (misma ciudad) → calcular distancia de conducción cuando sea posible
        s = near.site["location"]
        try:
            driving_km, _ = await driving_distance_km(client, s["lat"], s["long"], dlat, dlng, language=language)
            near.driving_distance_km = round(driving_km, 2)
            distance_km = driving_km
        except Exception:
            # Fallback Haversine (no conducido, solo para no romper)
            distance_km = near.distance_km

        # Redondeo de reporte (solo visual)
        distance_km_report = round(distance_km, DISTANCE_REPORT_DECIMALS)

        # 5) Calcular costo por tramos (SE MANTIENE TAL CUAL, usando KM)
        delivery_cost = compute_delivery_cost_cop(distance_km)

        # 6) (Opcional) Consultar metadatos de disponibilidad en Shipday — NO afecta el precio
        sd = None
        sd_payload = None
        sd_raw = None
        sd_ts = None
        pickup_duration_minutes = None
        delivery_duration_minutes = None
        pickup_time_iso = None
        delivery_time_iso_out = None

        try:
            if SHIPDAY_API_KEY_COLOMBIA:
                pickup_addr_str = near.site["site_address"]                    # tal como está registrada
                delivery_addr_str = raw_address or formatted                   # limpio y estable
                sd, sd_payload, sd_raw, sd_ts = await shipday_quote_meta_by_address(
                    client=client,
                    pickup_address=pickup_addr_str,
                    delivery_address=delivery_addr_str,
                    delivery_time_iso=delivery_time_iso,
                )
                if isinstance(sd, ShipdayAvailability):
                    pickup_duration_minutes = sd.pickup_duration_minutes
                    delivery_duration_minutes = sd.delivery_duration_minutes
                    pickup_time_iso = sd.pickup_time_iso
                    delivery_time_iso_out = sd.delivery_time_iso
        except Exception:
            # No bloquear si Shipday falla; devolvemos igual la cobertura y el precio
            pass

    return CoverageDetailsResponse(
        place_id=place_id,
        formatted_address=formatted,
        lat=dlat,
        lng=dlng,
        nearest=near,                          # sede asignada más cercana en la ciudad
        delivery_cost_cop=delivery_cost,       # costo por tramos
        distance_miles=distance_km_report,        # distancia (km) reportada
        dropoff=dropoff,
        # ===== Metadatos/auditoría Shipday (opcionales) =====
        pickup_duration_minutes=pickup_duration_minutes,
        delivery_duration_minutes=delivery_duration_minutes,
        pickup_time_iso=pickup_time_iso,
        delivery_time_iso=delivery_time_iso_out,
        shipday_payload=sd_payload,
        shipday_response=sd_raw,
        shipday_requested_at_iso=sd_ts,
        error=None
    )

@router.get("/places/details", response_model=GeocodedPoint)
async def places_details_endpoint(
    place_id: str = Query(..., description="Place ID a resolver"),
    session_token: Optional[str] = Query(None, description="Token de sesión usado en Autocomplete")
):
    async with httpx.AsyncClient() as client:
        lat, lng, formatted = await places_details(client, place_id, session_token)
    return GeocodedPoint(query=place_id, formatted_address=formatted, lat=lat, lng=lng)

@router.post("/distance", response_model=DistanceResponse)
async def compute_distance(
    body: DistanceRequest,
    method: str = Query("driving", pattern=r"^(haversine|driving)$")
):
    """
    Calcula distancia entre origen y destino.
    - 'driving' (por defecto): usa Google Distance Matrix → distance_km y distance_miles_driving.
    - 'haversine' : sólo para fallback; reporta distance_km y NO incluye miles (porque no son conducidas).
    """
    # Resolver origen/destino
    async with httpx.AsyncClient() as client:
        if body.origin_place_id:
            olat, olng, ofmt = await places_details(client, body.origin_place_id, body.session_token)
            oquery = body.origin_place_id
        elif body.origin:
            olat, olng, ofmt = await geocode_address(client, body.origin, countries="co")
            oquery = body.origin
        else:
            raise HTTPException(status_code=422, detail="Debes enviar origin o origin_place_id")

        if body.destination_place_id:
            dlat, dlng, dfmt = await places_details(client, body.destination_place_id, body.session_token)
            dquery = body.destination_place_id
        elif body.destination:
            dlat, dlng, dfmt = await geocode_address(client, body.destination, countries="co")
            dquery = body.destination
        else:
            raise HTTPException(status_code=422, detail="Debes enviar destination o destination_place_id")

    # Calcular distancia según método
    distance_km = 0.0
    distance_miles_driving: Optional[float] = None
    if method == "driving":
        async with httpx.AsyncClient() as client:
            km, _secs = await driving_distance_km(client, olat, olng, dlat, dlng)
        distance_km = km
        distance_miles_driving = round(km / 1.609344, 2)
        used_method = "google_distance_matrix_driving"
    else:
        km = haversine_km(olat, olng, dlat, dlng)
        distance_km = km
        used_method = "great_circle_haversine"

    origin_gc = GeocodedPoint(query=oquery, formatted_address=ofmt, lat=olat, lng=olng)
    destination_gc = GeocodedPoint(query=dquery, formatted_address=dfmt, lat=dlat, lng=dlng)

    return DistanceResponse(
        origin=origin_gc,
        destination=destination_gc,
        distance_km=round(distance_km, 2),
        distance_miles_driving=distance_miles_driving,
        method=used_method,
    )

# ======= ENDPOINT: distancia desde Shipday (pickup = sede asignada en misma ciudad) =======
@router.get("/shipday/distance", response_model=ShipdayDistanceResponse)
async def shipday_distance_from_nearest_site(
    place_id: str = Query(..., description="Place ID del destino (cliente)"),
    session_token: Optional[str] = Query(None, description="Token de sesión de Autocomplete"),
    language: str = Query("es", description="Idioma (no afecta Shipday)"),
):
    """
    Usa la sede más cercana dentro de la MISMA CIUDAD (tu cobertura) y consulta a Shipday
    la distancia de conducción entre esa sede (pickup) y el destino (dropoff).
    - Respuesta SIEMPRE en KM; adicionalmente, 'shipday_distance_miles_driving' derivada (km→mi).
    - No calcula precio (sólo distancia de Shipday).
    """
    if not SHIPDAY_API_KEY_COLOMBIA:
        raise HTTPException(status_code=500, detail="No se configuró SHIPDAY_API_KEY_COLOMBIA")

    async with httpx.AsyncClient() as client:
        # Asegurar sedes con coords y city_norm
        await ensure_sites_coords(client)

        # Resolver destino y componer 'dropoff'
        dlat, dlng, formatted, comps = await places_details_with_components(client, place_id, session_token)
        addr, raw_address = _build_address_from_components(comps)
        drop_city_norm = normalize_city(addr.city)

        # Elegir sede dentro de la misma ciudad (tu regla de cobertura)
        near = nearest_among_same_city(dlat, dlng, drop_city_norm)
        if not near.in_coverage or not near.site:
            return ShipdayDistanceResponse(
                place_id=place_id,
                formatted_address=formatted,
                lat=dlat,
                lng=dlng,
                nearest=None,
                error=make_out_of_coverage_error_by_city(addr.city)
            )

        pickup_addr_str = near.site["site_address"]  # dirección tal cual la tienes registrada en Shipday
        dropoff_addr_str = raw_address or formatted  # mejor enviar el dropoff “limpio”

        # Probar distancia en Shipday (pedido efímero)
        try:
            dist_raw, dist_km = await _shipday_probe_distance(
                client=client,
                pickup_address_str=pickup_addr_str,
                dropoff_address_str=dropoff_addr_str,
                pickup_name=near.site.get("site_name", "Pickup")
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Fallo consultando distancia en Shipday: {e}")

    return ShipdayDistanceResponse(
        place_id=place_id,
        formatted_address=formatted,
        lat=dlat,
        lng=dlng,
        nearest=near,
        shipday_distance_km=round(dist_km, 3),                         # km siempre
        shipday_distance_miles_driving=round(dist_km / 1.609344, 3),   # millas conducidas derivadas
        error=None
    )
