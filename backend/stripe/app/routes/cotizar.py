from __future__ import annotations

from fastapi import HTTPException, APIRouter, Query
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional, Dict, Any
import os, math, asyncio, uuid
import httpx
from dotenv import load_dotenv
import re
from datetime import datetime, timezone
import requests

load_dotenv()

# ─────────── Config / Constantes externas ───────────
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️  Falta GOOGLE_MAPS_API_KEY en el entorno (.env)")

PLACES_AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
PLACES_DETAILS_URL      = "https://maps.googleapis.com/maps/api/place/details/json"
GEOCODE_URL             = "https://maps.googleapis.com/maps/api/geocode/json"
DISTANCE_MATRIX_URL     = "https://maps.googleapis.com/maps/api/distancematrix/json"

# Shipday (ya no se usa para cotizar; se mantiene estructura por compatibilidad)
SHIPDAY_API_KEY_US = (
    os.getenv("SHIPDAY_API_KEY")
    or os.getenv("SHIPDAY_APIKEY")
)
SHIPDAY_API_KEY_COLOMBIA = (
    os.getenv("SHIPDAY_API_KEY_COLOMBIA")
    or os.getenv("SHIPDAY_APIKEY_COLOMBIA")
)

# NO consultar Shipday
SHIPDAY_ENABLED = False

SHIPDAY_AVAILABILITY_URL = "https://api.shipday.com/on-demand/availability"

# Fallback manual (USD por milla)
DELIVERY_RATE_USD_PER_MILE = float(os.getenv("DELIVERY_RATE_USD_PER_MILE", "2.0"))
DELIVERY_MIN_USD_OUT_OF_COVERAGE = float(os.getenv("DELIVERY_MIN_USD_OUT_OF_COVERAGE", "6.0"))

# Decimales para reportar distancia
DISTANCE_REPORT_DECIMALS = int(os.getenv("DISTANCE_REPORT_DECIMALS", "2"))
# Países permitidos para Places
PLACES_COUNTRIES = os.getenv("PLACES_COUNTRIES", "us")

# Límite de distancia por conducción (millas) — solo informativo
DRIVING_DISTANCE_MAX_MILES = float(os.getenv("DRIVING_DISTANCE_MAX_MILES", "8"))

# Proveedor preferido/secundario (no se usan ya)
SHIPDAY_PREFERRED_PROVIDER = (os.getenv("SHIPDAY_PREFERRED_PROVIDER", "doordash")).strip().lower()
SHIPDAY_SECONDARY_PROVIDER = (os.getenv("SHIPDAY_SECONDARY_PROVIDER", "uber")).strip().lower()

# Sitios que cuentan como NEWARK para mínimo 13 USD
NEWARK_SITE_IDS = {36}

# Mínimos cotizador manual
DELIVERY_MIN_USD_FALLBACK = float(os.getenv("DELIVERY_MIN_USD_FALLBACK", "6.0"))
NEWARK_MIN_USD            = float(os.getenv("NEWARK_MIN_USD", "13.0"))

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
    {
        "site_id": 37,
        "site_name": "NEW YORK",
        "site_address": "84-19 Northern Blvd L1, Jackson Heights, NY 11372",
        "site_phone": "+13477929350",
        "site_business_hours": "string",
        "horario_semanal": {
            "dia": "string"
        },
        "wsp_link": "https://wa.link/6v0usl",
        "city_id": 15,
        "maps": "https://share.google/xpM5tbzW3CZCtvLK9",
        "show_on_web": True,
        "email_address": "Hola@gmail.com",
        "status": True,
        "comming_soon": False,
        "open": True,
        "pe_site_id": 16,
        "exist": True,
        "invoice_identifier": "NEK",
        "time_zone": "America/New_York",
        "restaurant_id": 1,
        "city_name": "USA",
        "location": {"lat": 40.7556, "long": -73.8821},
        "pickup": {
            "address": {
                "zip": "11372",
                "city": "Jackson Heights",
                "unit": None,
                "state": "NY",
                "street": "84-19 Northern Blvd L1",
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
    distance_miles: float
    in_coverage: bool
    driving_distance_miles: Optional[float] = None

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

class Dropoff(BaseModel):
    address: Address

class CoverageDetailsResponse(BaseModel):
    place_id: str
    formatted_address: str
    lat: float
    lng: float
    nearest: Optional[NearestInfo] = None
    delivery_cost_cop: Optional[int] = None  # (en USD; nombre por compat)
    distance_miles: Optional[float] = None
    dropoff: Optional[Dropoff] = None
    pickup_duration_minutes: Optional[int] = None
    delivery_duration_minutes: Optional[int] = None
    pickup_time_iso: Optional[str] = None
    delivery_time_iso: Optional[str] = None
    shipday_payload: Optional[Dict[str, Any]] = None
    shipday_response: Optional[Any] = None
    shipday_requested_at_iso: Optional[str] = None
    error: Optional[CoverageError] = None

class ShipdayAvailability(BaseModel):
    provider: str
    fee: float
    pickup_duration_minutes: Optional[int] = None
    delivery_duration_minutes: Optional[int] = None
    pickup_time_iso: Optional[str] = None
    delivery_time_iso: Optional[str] = None

# ─────────── Lista negra de ubicaciones (ahora vacía/optativa) ───────────
BLACKLIST_LOCATIONS: List[Dict[str, Any]] = []

def _cf(s: Optional[str]) -> str:
    return (s or "").strip().casefold()

def _is_blacklisted_address(addr: Optional[Address], strict_label: str) -> Optional[str]:
    label_cf = _cf(strict_label)
    for rule in BLACKLIST_LOCATIONS:
        name = rule.get("name") or "ZONA BLOQUEADA"
        for frag in (rule.get("contains") or []):
            if frag and frag in label_cf:
                return name
        if addr:
            state_ok = True
            if rule.get("state"):
                state_ok = _cf(addr.state) == _cf(rule["state"])
            if not state_ok:
                continue
            if rule.get("city") and _cf(addr.city) == _cf(rule["city"]):
                return name
            if rule.get("city_in"):
                cities = [_cf(c) for c in rule["city_in"]]
                if _cf(addr.city) in cities:
                    return name
            if rule.get("zip_prefixes"):
                z = _cf(addr.zip)
                for pref in rule["zip_prefixes"]:
                    if z.startswith(_cf(pref)):
                        return name
    return None

def _is_blacklisted_string(address_str: str) -> Optional[str]:
    return _is_blacklisted_address(None, address_str)

# ─────────── Utilidades generales ───────────
def _normalize_spaces(txt: str) -> str:
    return re.sub(r"\s+", " ", txt or "").strip()

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

def _join_nonempty(parts: List[str], sep: str = ", ") -> str:
    return sep.join([p for p in parts if p and str(p).strip()])

_STREET_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")

def _format_address_strict(addr: Address) -> str:
    line2 = " ".join([p for p in [addr.state, addr.zip] if p])
    return ", ".join([addr.street, addr.city, line2, addr.country])

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
    return _format_address_strict(addr)

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
    return NearestInfo(site=best, distance_miles=round(best_dist, 2), in_coverage=True)

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

ZIP_TO_SITE: Dict[str, int] = {
    # Queens - Jackson Heights / alrededores
    "11369": 37, "11370": 37, "11371": 37,
    "11372": 37, "11373": 37, "11374": 37,
    "11375": 37, "11377": 37, "11378": 37,
    "11101": 37, "11102": 37, "11103": 37, "11104": 37, "11105": 37, "11106": 37,
    # Agrega más ZIPs según necesidad
}

def assign_site_by_zip(zip_code: str) -> Optional[Dict[str, Any]]:
    z = (zip_code or "").strip()
    if not z:
        return None
    sid = ZIP_TO_SITE.get(z)
    if sid is not None:
        for s in SEDES:
            if s.get("site_id") == sid:
                return s
    return None

# Geocoding simple (sin validaciones estrictas)
async def geocode_address(client: httpx.AsyncClient, address: str) -> Tuple[float, float, str]:
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
    formatted = top.get("formatted_address") or address
    return float(loc["lat"]), float(loc["lng"]), formatted

async def places_details(
    client: httpx.AsyncClient,
    place_id: str,
    session_token: Optional[str] = None
) -> Tuple[float, float, str]:
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
    formatted = res.get("formatted_address") or place_id
    return float(loc["lat"]), float(loc["lng"]), formatted

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
    formatted = res.get("formatted_address") or place_id
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

# ─────────── Helpers de pricing manual ───────────
def _is_newark_site(site: Optional[Dict[str, Any]]) -> bool:
    if not site:
        return False
    name = (site.get("site_name") or "").strip().lower()
    sid = site.get("site_id")
    return name == "newark" or sid in NEWARK_SITE_IDS

def manual_quote_usd(miles: float, pickup_site: Optional[Dict[str, Any]]) -> int:
    base = max(float(miles) * DELIVERY_RATE_USD_PER_MILE, DELIVERY_MIN_USD_FALLBACK)
    if _is_newark_site(pickup_site) and base < NEWARK_MIN_USD:
        base = NEWARK_MIN_USD
    return int(math.ceil(base))

# ─────────── Endpoints ───────────

# Nota: Este endpoint ahora solo informa que Shipday no está en uso
@router.post("/shipday/availability/{order_id}", response_model=Dict[str, Any])
async def shipday_availability(order_id: str):
    ts_iso = datetime.now(timezone.utc).isoformat()
    return {
        "available": False,
        "availability": None,
        "shipday_payload": None,
        "shipday_response": None,
        "shipday_requested_at_iso": ts_iso,
        "error": {
            "code": "SHIPDAY_DISABLED",
            "message_es": "Shipday no se consulta en esta configuración.",
            "message_en": "Shipday is not used in this configuration.",
        },
    }

# ===== Autocomplete SIN validaciones (rápido) =====
@router.get("/places/autocomplete", response_model=AutocompleteLiteResponse)
async def places_autocomplete(
    input: str = Query(..., min_length=1, description="Texto parcial de dirección"),
    session_token: Optional[str] = Query(None, description="Token de sesión para mejor facturación/resultados"),
    language: str = Query("es", description="Idioma de resultados"),
    countries: Optional[str] = Query(None, description="Códigos ISO-3166-1 alpha-2 separados por | o , (p.ej. 'us|pr')"),
    limit: int = Query(5, ge=1, le=10, description="Máximo de sugerencias"),
    strict: bool = Query(False, description="(Ignorado) Si True haría validación extra; ahora siempre rápido")
):
    """
    Devuelve predicciones de Autocomplete directamente SIN validar con Place Details para cada una.
    Mucho más rápido y barato (1 sola llamada).
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")

    stoken = session_token or str(uuid.uuid4())
    comps = _components_for(countries)

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

        raw_preds = (data.get("predictions") or [])[:limit]
        items = [
            AutocompleteLiteItem(
                description=p.get("description", ""),
                place_id=p.get("place_id", ""),
                types=p.get("types", []) or [],
            )
            for p in raw_preds
        ]
        return AutocompleteLiteResponse(predictions=items, session_token=stoken)

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
    coverage_radius_miles: float = Query(1000.0, gt=0, description="Radio de cobertura en millas (solo informativo)"),
    language: str = Query("es", description="Idioma para Distance Matrix"),
    delivery_time_iso: Optional[str] = Query(None, description="(Ignorado) Hora de entrega ISO-8601 (UTC)")
):
    """
    Cobertura + costo:
    - Usa Place Details UNA sola vez (para lat/lng + address_components).
    - NO valida que la dirección tenga número/zip/país (evita rechazos).
    - NO consulta Shipday; siempre usa cotizador manual.
    - Respeta ZIP_TO_SITE (fuerza sede), y ya NO se sobreescribe luego.
    """
    async with httpx.AsyncClient() as client:
        dlat, dlng, formatted, comps = await places_details_with_components(client, place_id, session_token)

        # Construye Address si es posible con los componentes; si falta algo, seguimos igual
        def _find_component(comps: List[Dict[str, Any]], type_name: str, short: bool = True) -> str:
            for c in comps:
                if type_name in (c.get("types") or []):
                    return c.get("short_name") if short and c.get("short_name") else c.get("long_name", "")
            return ""

        street_number = _find_component(comps, "street_number", short=False)
        route = _find_component(comps, "route", short=False)
        city = (
            _find_component(comps, "locality", short=False)
            or _find_component(comps, "sublocality", short=False)
            or _find_component(comps, "administrative_area_level_2", short=False)
            or ""
        )
        state = _find_component(comps, "administrative_area_level_1", short=True) or ""
        zip_code = _find_component(comps, "postal_code", short=False) or ""
        country = _find_component(comps, "country", short=True) or ""
        unit = _find_component(comps, "subpremise", short=False) or None
        street = " ".join([p for p in [street_number, route] if p]).strip()

        address = Address(
            unit=unit or None,
            street=street or formatted.split(",")[0],
            city=city or (formatted.split(",")[1].strip() if "," in formatted else ""),
            state=state,
            zip=zip_code,
            country=country or (formatted.split(",")[-1].strip() if "," in formatted else ""),
        )

        strict_label = _format_address_strict(address)
        dropoff = Dropoff(address=address)

        # Blacklist opcional (actualmente vacía)
        reason_blk = _is_blacklisted_address(address, strict_label)
        if reason_blk:
            near_blk = nearest_site_for(dlat, dlng, radius_miles=coverage_radius_miles)
            near_blk.in_coverage = False
            return CoverageDetailsResponse(
                place_id=place_id,
                formatted_address=strict_label,
                lat=dlat,
                lng=dlng,
                nearest=near_blk,
                delivery_cost_cop=None,
                distance_miles=None,
                dropoff=dropoff,
                pickup_duration_minutes=None,
                delivery_duration_minutes=None,
                pickup_time_iso=None,
                delivery_time_iso=None,
                shipday_payload=None,
                shipday_response=None,
                shipday_requested_at_iso=None,
                error=CoverageError(
                    code="OUT_OF_COVERAGE_BLACKLIST",
                    message_es="Fuera de cobertura, pronto abriremos aquí.",
                    message_en="Out of coverage, we will open here soon",
                ),
            )

        # 1) Intentar forzar sede por ZIP
        forced_site = assign_site_by_zip(address.zip)

        if forced_site:
            # Distancia haversine al forzado
            forced_dist = haversine_miles(forced_site["location"]["lat"], forced_site["location"]["long"], dlat, dlng)
            near = NearestInfo(site=forced_site, distance_miles=round(forced_dist, 2), in_coverage=True)
        else:
            # 2) Fallback a "nearest" geométrico si no hay ZIP forzado
            near = nearest_site_for(dlat, dlng, radius_miles=coverage_radius_miles)

        # Distancia por conducción (con fallback a haversine si falla)
        s = near.site["location"]
        try:
            driving_miles, _ = await driving_distance_miles(
                client, s["lat"], s["long"], dlat, dlng, language=language
            )
            near.driving_distance_miles = round(driving_miles, 2)
            d_miles = driving_miles
        except Exception:
            d_miles = near.distance_miles
            near.driving_distance_miles = None

        distance_miles_report = round(float(d_miles), DISTANCE_REPORT_DECIMALS)

        # Costo SIEMPRE manual (Shipday deshabilitado)
        cost_int = manual_quote_usd(float(d_miles), near.site)

    return CoverageDetailsResponse(
        place_id=place_id,
        formatted_address=strict_label,
        lat=dlat,
        lng=dlng,
        nearest=near,
        delivery_cost_cop=cost_int,
        distance_miles=distance_miles_report,
        dropoff=dropoff,
        pickup_duration_minutes=None,
        delivery_duration_minutes=None,
        pickup_time_iso=None,
        delivery_time_iso=None,
        shipday_payload=None,
        shipday_response=None,
        shipday_requested_at_iso=None,
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
