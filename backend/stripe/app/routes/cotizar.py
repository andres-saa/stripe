from __future__ import annotations

from fastapi import HTTPException, APIRouter, Query
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional, Dict, Any
import os, math, asyncio, uuid
import httpx
from dotenv import load_dotenv
import re

load_dotenv()

# ─────────── Config / Constantes externas ───────────
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  Falta GOOGLE_MAPS_API_KEY en el entorno (.env)")

PLACES_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
PLACES_DETAILS_URL      = "https://maps.googleapis.com/maps/api/place/details/json"
GEOCODE_URL             = "https://maps.googleapis.com/maps/api/geocode/json"
DISTANCE_MATRIX_URL     = "https://maps.googleapis.com/maps/api/distancematrix/json"

# Shipday
SHIPDAY_API_KEY = os.getenv("SHIPDAY_API_KEY")
if not SHIPDAY_API_KEY:
    print("⚠️  Falta SHIPDAY_API_KEY en el entorno (.env)")
SHIPDAY_AVAILABILITY_URL = "https://api.shipday.com/on-demand/availability"

# Fallback si Shipday no devuelve tarifa (USD por milla)
DELIVERY_RATE_USD_PER_MILE = float(os.getenv("DELIVERY_RATE_USD_PER_MILE", "1.5"))
# Decimales para reportar distancia
DISTANCE_REPORT_DECIMALS = int(os.getenv("DISTANCE_REPORT_DECIMALS", "2"))
# Países permitidos para Places
PLACES_COUNTRIES = os.getenv("PLACES_COUNTRIES", "us")  # p.ej. "us" o "us|pr"
# Límite de cobertura por distancia conducida (mi)
MAX_DRIVING_DISTANCE_MILES = float(os.getenv("MAX_DRIVING_DISTANCE_MILES", "15"))

router = APIRouter()

# ─────────── SEDES (USA) ───────────
SEDES: List[Dict[str, Any]] = [
    {
        "site_id": 33,
        "site_name": "UNION CITY",
        "site_address": "2100 kerrigan ave union city nj 07087",
        "pe_site_id": 16,
        "location": {"lat": 40.76808, "long": -74.03843},
        "pickup": {
            "address": {
                "zip": "07087",
                "city": "Union City",
                "unit": None,
                "state": "NJ",
                "street": "2100 Kerrigan Ave",
                "country": "US",
            }
        },
    },
    {
        "site_id": 35,
        "site_name": "FILADELPHIA",
        "site_address": "5759 Oxford ave, Philadelphia, PA 19149",
        "pe_site_id": 16,
        "location": {"lat": 40.03358, "long": -75.08501},
        "pickup": {
            "address": {
                "zip": "19149",
                "city": "Philadelphia",
                "unit": None,
                "state": "PA",
                "street": "5759 Oxford Ave",
                "country": "US",
            }
        },
    },
    {
        "site_id": 36,
        "site_name": "NEWARK",
        "site_address": "2100 kerrigan ave union city nj 07087",
        "pe_site_id": 16,
        "location": {"lat": 40.71335, "long": -74.20744},
        "pickup": {
            "address": {
                "zip": "07087",
                "city": "Union City",
                "unit": None,
                "state": "NJ",
                "street": "2100 Kerrigan Ave",
                "country": "US",
            }
        },
    },
]

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
    distance_miles: float
    distance_km: float
    method: str = "great_circle_haversine"

class NearestInfo(BaseModel):
    site: Dict[str, Any]
    distance_miles: float              # Haversine (rápida)
    in_coverage: bool
    driving_distance_miles: Optional[float] = None  # por carretera (Distance Matrix)

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
    unit: Optional[str] = None          # Apt/Suite/etc (subpremise)
    street: str                         # "<street_number> <route>"
    city: str
    state: str                          # e.g. "NJ"
    zip: str                            # postal_code
    country: str

class Dropoff(BaseModel):
    address: Address

class CoverageDetailsResponse(BaseModel):
    place_id: str
    formatted_address: str
    lat: float
    lng: float
    nearest: Optional[NearestInfo] = None
    delivery_cost_cop: Optional[int] = None   # (en USD si viene de Shipday; nombre por compat)
    distance_miles: Optional[float] = None
    dropoff: Optional[Dropoff] = None
    pickup_duration_minutes: Optional[int] = None     # ← pickupDuration
    delivery_duration_minutes: Optional[int] = None   # ← deliveryDuration
    pickup_time_iso: Optional[str] = None             # ← pickupTime
    delivery_time_iso: Optional[str] = None           # ← deliveryTime
    error: Optional[CoverageError] = None

class ShipdayAvailability(BaseModel):
    provider: str
    fee: float
    pickup_duration_minutes: Optional[int] = None
    delivery_duration_minutes: Optional[int] = None
    pickup_time_iso: Optional[str] = None
    delivery_time_iso: Optional[str] = None

# ─────────── Utilidades generales ───────────

HOUSE_NUM_SEEDS = [100, 200, 500, 800, 1000, 1500, 2000, 2100, 2500]
STREET_SUFFIXES = ["Ave", "Avenue", "St", "Street", "Rd", "Road", "Blvd", "Lane", "Dr"]

def _normalize_spaces(txt: str) -> str:
    return re.sub(r"\s+", " ", txt or "").strip()

def _looks_like_street_only(user_input: str) -> bool:
    s = _normalize_spaces(user_input)
    if not s or any(ch.isdigit() for ch in s):
        return False
    words = s.split()
    return len(words) <= 3

def _ensure_route_phrase(base_input: str) -> List[str]:
    s = _normalize_spaces(base_input)
    if not s:
        return []
    title = s.title()
    if any(re.search(rf"\b{sf}\b", title, flags=re.I) for sf in STREET_SUFFIXES):
        return [title]
    variants = [f"{title} {sf}" for sf in ["Ave", "St", "Road", "Blvd"]]
    seen, out = set(), []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out

def _cities_from_sedes() -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for s in SEDES:
        addr = (s.get("pickup") or {}).get("address") or {}
        city = (addr.get("city") or "").strip()
        state = (addr.get("state") or "").strip()
        if city and state:
            key = f"{city}|{state}"
            if key not in seen:
                seen.add(key)
                out.append({"city": city, "state": state})
    if not out:
        out = [{"city": "Union City", "state": "NJ"},
               {"city": "Newark", "state": "NJ"},
               {"city": "Philadelphia", "state": "PA"}]
    return out

def _city_center(city: str, state: str) -> Optional[Tuple[float, float]]:
    for s in SEDES:
        a = (s.get("pickup") or {}).get("address") or {}
        if (a.get("city") or "").lower() == city.lower() and (a.get("state") or "").upper() == state.upper():
            loc = s.get("location") or {}
            if "lat" in loc and "long" in loc:
                return float(loc["lat"]), float(loc["long"])
    return None

def _join_nonempty(parts: List[str], sep: str = ", ") -> str:
    return sep.join([p for p in parts if p and str(p).strip()])

_STREET_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")  # exige número y nombre de vía

def _validate_address_required(addr: Address) -> List[str]:
    errors: List[str] = []
    if not addr.street or not _STREET_RE.match(addr.street):
        errors.append("street (debe incluir número y nombre de vía)")
    if not addr.city:
        errors.append("city")
    if not addr.state:
        errors.append("state")
    if not addr.zip:
        errors.append("zip")
    if not addr.country:
        errors.append("country (ISO-2)")
    return errors

def _format_address_strict(addr: Address) -> str:
    errs = _validate_address_required(addr)
    if errs:
        raise ValueError(", ".join(errs))
    line2 = " ".join([p for p in [addr.state, addr.zip] if p])
    return ", ".join([addr.street, addr.city, line2, addr.country])

def _address_to_str(addr: Address) -> str:
    return _format_address_strict(addr)

def _site_pickup_address_str(site: Dict[str, Any]) -> str:
    pickup_addr = (site.get("pickup") or {}).get("address") or {}
    if not pickup_addr:
        raise HTTPException(status_code=500, detail=f"sede '{site.get('site_name')}' sin pickup.address configurado")

    addr = Address(
        unit=pickup_addr.get("unit"),
        street=pickup_addr.get("street", ""),
        city=pickup_addr.get("city", ""),
        state=pickup_addr.get("state", ""),
        zip=pickup_addr.get("zip", ""),
        country=pickup_addr.get("country", ""),
    )
    try:
        return _format_address_strict(addr)
    except ValueError as ve:
        raise HTTPException(
            status_code=500,
            detail=f"sede '{site.get('site_name')}' pickup misconfigured: faltan campos -> {str(ve)}"
        )

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_miles * c

def nearest_site_for(lat: float, lng: float, radius_miles: float = 1000.0) -> NearestInfo:
    best = None
    best_dist = float("inf")
    for s in SEDES:
        slat = s["location"]["lat"]
        slng = s["location"]["long"]
        d = haversine_miles(lat, lng, slat, slng)
        if d < best_dist:
            best_dist = d
            best = s
    return NearestInfo(site=best, distance_miles=round(best_dist, 2), in_coverage=(best_dist <= radius_miles))

def make_out_of_coverage_error_by_city(city: str) -> CoverageError:
    city_txt = city or "la ciudad indicada"
    return CoverageError(
        code="OUT_OF_COVERAGE",
        message_es=f"No hay cobertura en {city_txt}. Aún no tenemos sedes en esa ciudad.",
        message_en=f"No coverage in {city_txt}. We don't have locations in that city yet.",
    )

# ─────────── Google APIs ───────────

def _components_for(countries: Optional[str] = None) -> Optional[str]:
    countries = countries or PLACES_COUNTRIES
    raw = [t.strip().lower() for t in re.split(r"[,\| ]+", countries) if t.strip()]

    alias = {
        "usa": "us", "u.s.a.": "us", "u.s.": "us", "eeuu": "us",
        "estados-unidos": "us", "estadosunidos": "us",
        "puerto-rico": "pr", "puertorico": "pr",
    }

    norm: List[str] = []
    for t in raw:
        t = alias.get(t, t)
        if re.fullmatch(r"[a-z]{2}", t):
            norm.append(t)

    norm = list(dict.fromkeys(norm))
    if not norm:
        return None
    return "|".join(f"country:{c}" for c in norm)

async def geocode_address(client: httpx.AsyncClient, address: str) -> Tuple[float, float, str]:
    """
    Geocodifica y devuelve SIEMPRE una etiqueta estricta: '123 Main St, City, ST 99999, US'.
    Falla con 400 si no se puede construir.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {"address": address, "key": GOOGLE_API_KEY, "components": _components_for()}
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
    comps = top.get("address_components", []) or []
    addr, _raw = _build_address_from_components(comps)
    errs = _validate_address_required(addr)
    if errs:
        raise HTTPException(status_code=400, detail=f"Dirección incompleta: faltan {', '.join(errs)}")
    strict_label = _address_to_str(addr)
    return float(loc["lat"]), float(loc["lng"]), strict_label

async def places_details(
    client: httpx.AsyncClient,
    place_id: str,
    session_token: Optional[str] = None
) -> Tuple[float, float, str]:
    """
    Resuelve SIEMPRE etiqueta estricta a partir de address_components.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "formatted_address,geometry,address_components"
    }
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
    comps = res.get("address_components", []) or []
    addr, _raw = _build_address_from_components(comps)
    errs = _validate_address_required(addr)
    if errs:
        raise HTTPException(status_code=400, detail=f"Dirección incompleta: faltan {', '.join(errs)}")
    strict_label = _address_to_str(addr)
    return float(loc["lat"]), float(loc["lng"]), strict_label

async def places_details_with_components(
    client: httpx.AsyncClient,
    place_id: str,
    session_token: Optional[str] = None
) -> Tuple[float, float, str, List[Dict[str, Any]]]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "formatted_address,geometry,address_components"
    }
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
    comps = res.get("address_components", []) or []
    # construir etiqueta estricta aquí también para reutilizar en coverage
    addr, _raw = _build_address_from_components(comps)
    errs = _validate_address_required(addr)
    if errs:
        # devolvemos formatted original pero ojo: los callers pueden decidir cómo manejarlo
        formatted = res.get("formatted_address", "") or place_id
    else:
        formatted = _address_to_str(addr)
    return float(loc["lat"]), float(loc["lng"]), (formatted or place_id), comps

async def driving_distance_miles(
    client: httpx.AsyncClient,
    o_lat: float, o_lng: float,
    d_lat: float, d_lng: float,
    language: str = "es"
) -> Tuple[float, int]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")

    params = {
        "origins": f"{o_lat},{o_lng}",
        "destinations": f"{d_lat},{d_lng}",
        "mode": "driving",
        "units": "imperial",
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
    miles   = meters / 1609.344
    return float(miles), int(seconds)

# ─────────── Shipday helpers ───────────

SHIPDAY_PREFERRED_PROVIDER = os.getenv("SHIPDAY_PREFERRED_PROVIDER", "uber").strip().lower()

def _shipday_auth_value() -> str:
    key = (SHIPDAY_API_KEY or "").strip()
    if not key:
        return ""
    return key if key.lower().startswith("basic ") else f"Basic {key}"

def _as_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        m = re.search(r"\d+", val)
        return int(m.group(0)) if m else None
    return None

class ShipdayAvailability(BaseModel):
    provider: str
    fee: float
    pickup_duration_minutes: Optional[int] = None
    delivery_duration_minutes: Optional[int] = None
    pickup_time_iso: Optional[str] = None
    delivery_time_iso: Optional[str] = None

async def shipday_quote_fee_by_address(
    client: httpx.AsyncClient,
    pickup_address: str,
    delivery_address: str,
    delivery_time_iso: Optional[str] = None
) -> Optional[ShipdayAvailability]:
    if not SHIPDAY_API_KEY:
        return None

    headers = {
        "Authorization": _shipday_auth_value(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: Dict[str, Any] = {
        "pickupAddress": pickup_address,
        "deliveryAddress": delivery_address,
    }
    if delivery_time_iso:
        payload["deliveryTime"] = delivery_time_iso

    try:
        resp = await client.post(SHIPDAY_AVAILABILITY_URL, headers=headers, json=payload, timeout=1000.0)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    if not isinstance(data, list):
        return None

    preferred = SHIPDAY_PREFERRED_PROVIDER

    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    def _provider_name(item: Dict[str, Any]) -> str:
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

    best: Optional[ShipdayAvailability] = None
    for item in data:
        if not item or item.get("error") is True:
            continue
        fee = item.get("fee")
        if not isinstance(fee, (int, float)):
            continue
        if preferred in _provider_name(item):
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

    return best

# Versión por coordenadas (compat)
async def shipday_quote_fee(
    client: httpx.AsyncClient,
    pickup_lat: float, pickup_lng: float,
    drop_lat: float, drop_lng: float
) -> Optional[float]:
    headers = {
        "Authorization": _shipday_auth_value(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "pickupLatitude": pickup_lat,
        "pickupLongitude": pickup_lng,
        "dropoffLatitude": drop_lat,
        "dropoffLongitude": drop_lng,
    }

    try:
        resp = await client.post(SHIPDAY_AVAILABILITY_URL, headers=headers, json=payload, timeout=1000.0)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    if not isinstance(data, list):
        return None

    preferred = SHIPDAY_PREFERRED_PROVIDER

    def _norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    def _provider_name(item: Dict[str, Any]) -> str:
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

    uber_fees: List[float] = []
    for item in data:
        if not item or item.get("error") is True:
            continue
        fee = item.get("fee")
        if not isinstance(fee, (int, float)):
            continue
        name = _provider_name(item)
        if preferred in name:
            uber_fees.append(float(fee))

    return min(uber_fees) if uber_fees else None

# ─────────── Helpers Autocomplete ───────────

async def _strict_label_from_place(
    client: httpx.AsyncClient,
    place_id: str,
    session_token: Optional[str]
) -> Optional[Tuple[Address, str]]:
    """
    Valida con Place Details y construye una etiqueta estricta (NUM STREET, CITY, ST ZIP, COUNTRY).
    Devuelve (Address, label) solo si tiene número, ciudad, estado, zip y país.
    """
    try:
        _lat, _lng, _formatted, comps = await places_details_with_components(
            client, place_id, session_token
        )
        addr, _raw = _build_address_from_components(comps)
        errs = _validate_address_required(addr)
        if errs:
            return None
        label = _address_to_str(addr)
        return addr, label
    except Exception:
        return None

FALLBACK_STREETS_US = [
    "Main St", "Market St", "Broadway",
    "First Ave", "Second St", "Third Ave",
    "Maple Ave", "Oak St", "Pine St",
    "Cedar St", "Washington St", "Church St",
    "Center St", "High St", "Elm St"
]

def _find_component(comps: List[Dict[str, Any]], type_name: str, short: bool = True) -> str:
    for c in comps:
        if type_name in (c.get("types") or []):
            return c.get("short_name") if short and c.get("short_name") else c.get("long_name", "")
    return ""

def _build_address_from_components(comps: List[Dict[str, Any]]) -> Tuple[Address, str]:
    street_number = _find_component(comps, "street_number", short=False)
    route = _find_component(comps, "route", short=False)
    street = f"{street_number} {route}".strip() if (street_number or route) else ""

    city = (
        _find_component(comps, "locality", short=False)
        or _find_component(comps, "sublocality", short=False)
        or _find_component(comps, "administrative_area_level_2", short=False)
        or ""
    )

    state = _find_component(comps, "administrative_area_level_1", short=True) or ""
    zip_code = _find_component(comps, "postal_code", short=False) or ""
    country = _find_component(comps, "country", short=True) or ""  # ISO-2
    unit = _find_component(comps, "subpremise", short=False) or None

    street = re.sub(r"\s+", " ", street).strip()
    city = re.sub(r"\s+", " ", city).strip()

    addr = Address(
        unit=unit,
        street=street,
        city=city,
        state=state,
        zip=zip_code,
        country=country
    )

    raw_address = ", ".join(
        filter(None, [street_number, route, city, state, zip_code, country])
    )
    return addr, raw_address

async def _fallback_address_suggestions(
    client: httpx.AsyncClient,
    base_input: str,
    stoken: str,
    language: str,
    comps: Optional[str],
    limit: int
) -> List[AutocompleteLiteItem]:
    results: List[AutocompleteLiteItem] = []
    seen_place_ids: set = set()
    seen_labels: set = set()

    route_candidates = _ensure_route_phrase(base_input) if _looks_like_street_only(base_input) else [base_input.title()]
    city_seeds = _cities_from_sedes()

    for city_item in city_seeds:
        if len(results) >= limit:
            break
        city = city_item["city"]
        state = city_item["state"]
        bias = _city_center(city, state)
        locationbias = f"circle:25000@{bias[0]},{bias[1]}" if bias else None  # ~25km

        for route in route_candidates:
            if len(results) >= limit:
                break

            for num in HOUSE_NUM_SEEDS:
                if len(results) >= limit:
                    break
                query = f"{num} {route}, {city}, {state}"

                params = {
                    "input": query,
                    "key": GOOGLE_API_KEY,
                    "language": language,
                    "sessiontoken": stoken,
                    "types": "address",
                }
                if comps:
                    params["components"] = comps
                if locationbias:
                    params["locationbias"] = locationbias

                try:
                    resp = await client.get(PLACES_AUTOCOMPLETE_URL, params=params, timeout=8.0)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    if data.get("status") != "OK":
                        continue
                    preds = (data.get("predictions") or [])[:3]
                except Exception:
                    continue

                tasks = [
                    _strict_label_from_place(client, p.get("place_id", ""), stoken)
                    for p in preds if p.get("place_id")
                ]
                details = await asyncio.gather(*tasks, return_exceptions=True)

                for p, det in zip(preds, details):
                    if isinstance(det, Exception) or not det:
                        continue
                    _addr, label = det
                    pid = p.get("place_id", "")
                    if not pid or pid in seen_place_ids or label in seen_labels:
                        continue
                    seen_place_ids.add(pid)
                    seen_labels.add(label)
                    results.append(
                        AutocompleteLiteItem(
                            description=label,
                            place_id=pid,
                            types=p.get("types", []) or [],
                        )
                    )
                    if len(results) >= limit:
                        break
    return results

# ─────────── Endpoints ───────────

@router.get("/places/autocomplete", response_model=AutocompleteLiteResponse)
async def places_autocomplete(
    input: str = Query(..., min_length=1, description="Texto parcial de dirección"),
    session_token: Optional[str] = Query(None, description="Token de sesión para mejor facturación/resultados"),
    language: str = Query("es", description="Idioma de resultados"),
    countries: Optional[str] = Query(None, description="Códigos ISO-3166-1 alpha-2 separados por | o , (p.ej. 'us|pr')"),
    limit: int = Query(5, ge=1, le=10, description="Máximo de sugerencias"),
    strict: bool = Query(True, description="Solo sugerencias bien formadas (número, calle, ciudad, estado, ZIP, país)")
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")

    stoken = session_token or str(uuid.uuid4())
    comps = _components_for(countries)

    oversample_factor = 3
    fetch_n = min(20, max(limit, limit * oversample_factor))

    params = {
        "input": input,
        "key": GOOGLE_API_KEY,
        "language": language,
        "sessiontoken": stoken,
        "types": "address",
    }
    if comps:
        params["components"] = comps

    async with httpx.AsyncClient() as client:
        resp = await client.get(PLACES_AUTOCOMPLETE_URL, params=params, timeout=15.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Error en Autocomplete: HTTP {resp.status_code}")

        data = resp.json()
        status = data.get("status")

        if status not in ("OK", "ZERO_RESULTS"):
            msg = data.get("error_message") or status or "Error en Autocomplete"
            raise HTTPException(status_code=400, detail=msg)

        raw_preds = (data.get("predictions") or [])[:fetch_n]

        if not strict:
            items = [
                AutocompleteLiteItem(
                    description=p.get("description", ""),
                    place_id=p.get("place_id", ""),
                    types=p.get("types", []) or [],
                )
                for p in raw_preds[:limit]
            ]
            return AutocompleteLiteResponse(predictions=items, session_token=stoken)

        tasks = [
            _strict_label_from_place(client, p.get("place_id", ""), stoken)
            for p in raw_preds if p.get("place_id")
        ]
        details_results = await asyncio.gather(*tasks, return_exceptions=True)

        strict_items: List[AutocompleteLiteItem] = []
        seen_labels = set()
        for p, det in zip(raw_preds, details_results):
            if isinstance(det, Exception) or not det:
                continue
            _addr, strict_label = det
            if strict_label in seen_labels:
                continue
            seen_labels.add(strict_label)
            strict_items.append(
                AutocompleteLiteItem(
                    description=strict_label,
                    place_id=p.get("place_id", ""),
                    types=p.get("types", []) or [],
                )
            )
            if len(strict_items) >= limit:
                break

        if len(strict_items) < limit:
            fallback_items = await _fallback_address_suggestions(
                client=client,
                base_input=input,
                stoken=stoken,
                language=language,
                comps=comps,
                limit=limit - len(strict_items),
            )
            strict_items.extend(fallback_items)

        return AutocompleteLiteResponse(predictions=strict_items[:limit], session_token=stoken)

@router.get("/places/details", response_model=GeocodedPoint)
async def places_details_endpoint(
    place_id: str = Query(..., description="Place ID a resolver"),
    session_token: Optional[str] = Query(None, description="Token de sesión usado en Autocomplete")
):
    async with httpx.AsyncClient() as client:
        lat, lng, formatted = await places_details(client, place_id, session_token)
    return GeocodedPoint(query=place_id, formatted_address=formatted, lat=lat, lng=lng)

@router.get("/places/coverage-details", response_model=CoverageDetailsResponse)
async def coverage_details(
    place_id: str = Query(..., description="Place ID seleccionado por el usuario"),
    session_token: Optional[str] = Query(None, description="Token de sesión usado en Autocomplete"),
    coverage_radius_miles: float = Query(1000.0, gt=0, description="Radio de cobertura en millas"),
    language: str = Query("es", description="Idioma para Distance Matrix"),
    delivery_time_iso: Optional[str] = Query(None, description="Hora de entrega ISO-8601 (UTC), opcional para Shipday")
):
    async with httpx.AsyncClient() as client:
        dlat, dlng, formatted, comps = await places_details_with_components(client, place_id, session_token)

        address, _ = _build_address_from_components(comps)
        drop_errs = _validate_address_required(address)
        if drop_errs:
            return CoverageDetailsResponse(
                place_id=place_id,
                formatted_address=formatted,  # puede ser el formatted de Google si falló la estricta
                lat=dlat,
                lng=dlng,
                nearest=None,
                delivery_cost_cop=None,
                distance_miles=None,
                dropoff=None,
                pickup_duration_minutes=None,
                delivery_duration_minutes=None,
                pickup_time_iso=None,
                delivery_time_iso=None,
                error=CoverageError(
                    code="MALFORMED_ADDRESS",
                    message_es="Por favor selecciona una dirección exacta con número y código postal. "
                               "Formato requerido: '123 Main St, City, ST 99999, US'. "
                               f"Faltan: {', '.join(drop_errs)}",
                    message_en="Please select an exact address with street number and postal code. "
                               "Required format: '123 Main St, City, ST 99999, US'. "
                               f"Missing: {', '.join(drop_errs)}",
                ),
            )

        # Etiqueta estricta garantizada aquí
        strict_label = _address_to_str(address)
        dropoff = Dropoff(address=address)

        near = nearest_site_for(dlat, dlng, radius_miles=coverage_radius_miles)
        if not near.in_coverage:
            return CoverageDetailsResponse(
                place_id=place_id,
                formatted_address=strict_label,
                lat=dlat,
                lng=dlng,
                nearest=None,
                delivery_cost_cop=None,
                distance_miles=None,
                dropoff=dropoff,
                pickup_duration_minutes=None,
                delivery_duration_minutes=None,
                pickup_time_iso=None,
                delivery_time_iso=None,
                error=make_out_of_coverage_error_by_city(address.city),
            )

        # Distancia por conducción
        s = near.site["location"]
        try:
            driving_miles, _ = await driving_distance_miles(
                client, s["lat"], s["long"], dlat, dlng, language=language
            )
            near.driving_distance_miles = round(driving_miles, 2)
            d_miles = driving_miles
        except Exception:
            # fallback si falla Distance Matrix: usar haversine
            d_miles = near.distance_miles
            near.driving_distance_miles = None

        # ── Regla de cobertura por distancia CONDUCIDA ──
        if d_miles is not None and float(d_miles) > MAX_DRIVING_DISTANCE_MILES:
            return CoverageDetailsResponse(
                place_id=place_id,
                formatted_address=strict_label,
                lat=dlat,
                lng=dlng,
                nearest=near,
                delivery_cost_cop=None,
                distance_miles=None,  # mantenemos None en error para no romper front
                dropoff=dropoff,
                pickup_duration_minutes=None,
                delivery_duration_minutes=None,
                pickup_time_iso=None,
                delivery_time_iso=None,
                error=CoverageError(
                    code="OUT_OF_COVERAGE",
                    message_es=(
                        f"Fuera de cobertura: la distancia por conducción supera "
                        f"{MAX_DRIVING_DISTANCE_MILES} millas."
                    ),
                    message_en=(
                        f"Out of coverage: driving distance exceeds "
                        f"{MAX_DRIVING_DISTANCE_MILES} miles."
                    ),
                ),
            )

        distance_miles_report = round(float(d_miles), DISTANCE_REPORT_DECIMALS)

        pickup_addr_str   = _site_pickup_address_str(near.site)
        delivery_addr_str = strict_label  # ya en formato estricto

        # Intentar cotizar con Shipday (solo si está dentro de cobertura por distancia)
        sd = await shipday_quote_fee_by_address(
            client,
            pickup_address=pickup_addr_str,
            delivery_address=delivery_addr_str,
            delivery_time_iso=delivery_time_iso
        )

        pickup_duration_minutes: Optional[int] = None
        delivery_duration_minutes: Optional[int] = None
        pickup_time_iso: Optional[str] = None
        delivery_time_iso_out: Optional[str] = None

        if isinstance(sd, ShipdayAvailability):
            cost_int = int(math.ceil(float(sd.fee)))  # USD (nombre compat)
            pickup_duration_minutes = sd.pickup_duration_minutes
            delivery_duration_minutes = sd.delivery_duration_minutes
            pickup_time_iso = sd.pickup_time_iso
            delivery_time_iso_out = sd.delivery_time_iso
        else:
            miles_for_cost = near.driving_distance_miles or d_miles
            cost_int = int(math.ceil(max(round(miles_for_cost * DELIVERY_RATE_USD_PER_MILE, 2), DELIVERY_RATE_USD_PER_MILE)))

    return CoverageDetailsResponse(
        place_id=place_id,
        formatted_address=strict_label,
        lat=dlat,
        lng=dlng,
        nearest=near,
        delivery_cost_cop=cost_int,
        distance_miles=distance_miles_report,
        dropoff=dropoff,
        pickup_duration_minutes=pickup_duration_minutes,
        delivery_duration_minutes=delivery_duration_minutes,
        pickup_time_iso=pickup_time_iso,
        delivery_time_iso=delivery_time_iso_out,
        error=None
    )

@router.post("/distance", response_model=DistanceResponse)
async def compute_distance(
    body: DistanceRequest,
    method: str = Query("driving", pattern=r"^(haversine|driving)$")
):
    async with httpx.AsyncClient() as client:
        if body.origin_place_id:
            olat, olng, ofmt = await places_details(client, body.origin_place_id, body.session_token)
            oquery = body.origin_place_id
        elif body.origin:
            olat, olng, ofmt = await geocode_address(client, body.origin)
            oquery = body.origin
        else:
            raise HTTPException(status_code=422, detail="Debes enviar origin o origin_place_id")

        if body.destination_place_id:
            dlat, dlng, dfmt = await places_details(client, body.destination_place_id, body.session_token)
            dquery = body.destination_place_id
        elif body.destination:
            dlat, dlng, dfmt = await geocode_address(client, body.destination)
            dquery = body.destination
        else:
            raise HTTPException(status_code=422, detail="Debes enviar destination o destination_place_id")

    if method == "driving":
        async with httpx.AsyncClient() as client:
            miles, _secs = await driving_distance_miles(client, olat, olng, dlat, dlng)
        km = miles * 1.609344
        used_method = "google_distance_matrix_driving"
    else:
        miles = haversine_miles(olat, olng, dlat, dlng)
        km = miles * 1.609344
        used_method = "great_circle_haversine"

    origin_gc = GeocodedPoint(query=oquery, formatted_address=ofmt, lat=olat, lng=olng)
    destination_gc = GeocodedPoint(query=dquery, formatted_address=dfmt, lat=dlat, lng=dlng)

    return DistanceResponse(
        origin=origin_gc,
        destination=destination_gc,
        distance_miles=round(miles, 2),
        distance_km=round(km, 2),
        method=used_method,
    )
