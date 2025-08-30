from fastapi import HTTPException, APIRouter, Query
from pydantic import BaseModel, Field
from typing import Tuple, List, Optional, Dict, Any
import os, math, asyncio, uuid
import httpx
from dotenv import load_dotenv
import re

load_dotenv()

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

# Decimales para reportar distancia (km)
DISTANCE_REPORT_DECIMALS = int(os.getenv("DISTANCE_REPORT_DECIMALS", "2"))

router = APIRouter()

# ─────────── SEDES (USA) ───────────
from typing import List, Dict, Any

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
        "site_address": "26 eckert ave Newark NJ 07112",
        "pe_site_id": 16,
        "location": {"lat": 40.71335, "long": -74.20744},
        "pickup": {
            "address": {
                "zip": "07112",
                "city": "Newark",
                "unit": None,
                "state": "NJ",
                "street": "26 Eckert Ave",
                "country": "US",
            }
        },
    },
]


PLACES_COUNTRIES = os.getenv("PLACES_COUNTRIES", "us")  # p.ej. "us" o "us|pr"

def _components_for(countries: Optional[str] = None) -> Optional[str]:
    """
    Convierte 'us|pr' o 'usa, puerto-rico' -> 'country:us|country:pr'
    Filtra y normaliza a ISO-3166-1 alpha-2.
    """
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
        if re.fullmatch(r"[a-z]{2}", t):  # solo alpha-2
            norm.append(t)

    norm = list(dict.fromkeys(norm))
    if not norm:
        return None
    return "|".join(f"country:{c}" for c in norm)

# ─────────── Modelos (mismo shape que CO) ───────────

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
    address: Address  # <- SOLO address (igual que pides)

class CoverageDetailsResponse(BaseModel):
    place_id: str
    formatted_address: str
    lat: float
    lng: float
    nearest: Optional[NearestInfo] = None
    delivery_cost_cop: Optional[int] = None   # <- MISMO NOMBRE (aunque el valor es USD)
    distance_km: Optional[float] = None       # <- MISMO NOMBRE
    dropoff: Optional[Dropoff] = None
    error: Optional[CoverageError] = None

# ─────────── Utilidades ───────────

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
    formatted = top.get("formatted_address", address)
    return float(loc["lat"]), float(loc["lng"]), formatted

async def places_details(
    client: httpx.AsyncClient,
    place_id: str,
    session_token: Optional[str] = None
) -> Tuple[float, float, str]:
    """
    Versión ligera: NO devuelve components (se usa en otros endpoints ya existentes).
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "formatted_address,geometry/location"
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
    formatted = res.get("formatted_address", "")
    return float(loc["lat"]), float(loc["lng"]), formatted or place_id

async def places_details_with_components(
    client: httpx.AsyncClient,
    place_id: str,
    session_token: Optional[str] = None
) -> Tuple[float, float, str, List[Dict[str, Any]]]:
    """
    Versión extendida: incluye address_components para construir dropoff.address.
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "formatted_address,geometry,address_component"
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
    formatted = res.get("formatted_address", "")
    comps = res.get("address_components", []) or []
    return float(loc["lat"]), float(loc["lng"]), (formatted or place_id), comps

def _find_component(comps: List[Dict[str, Any]], type_name: str, short: bool = True) -> str:
    for c in comps:
        if type_name in (c.get("types") or []):
            # country/state conviene short_name (US, NJ)
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

    # Limpieza
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

    # Siempre en formato ST_NUM, ST_NAME, CITY, STATE, ZIP, COUNTRY
    raw_address = ", ".join(
        filter(None, [street_number, route, city, state, zip_code, country])
    )

    return addr, raw_address

async def driving_distance_miles(
    client: httpx.AsyncClient,
    o_lat: float, o_lng: float,
    d_lat: float, d_lng: float,
    language: str = "es"
) -> Tuple[float, int]:
    """
    Devuelve (millas_por_carretera, duracion_segundos) usando Google Distance Matrix.
    """
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

# ─────────── SHIPDAY (Availability) ───────────

async def shipday_quote_fee(
    client: httpx.AsyncClient,
    pickup_lat: float, pickup_lng: float,
    drop_lat: float, drop_lng: float
) -> Optional[float]:
    """
    Pide a Shipday disponibilidad/tarifa entre (pickup) y (drop).
    Devuelve la tarifa mínima (fee) entre los servicios disponibles, o None si no hay tarifa.
    """
    headers = {
        "Authorization": SHIPDAY_API_KEY,   # On-Demand Availability usa este header
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
        # 400 típico: "no third party delivery service has been enabled by the user"
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    if not isinstance(data, list):
        return None

    fees = []
    for item in data:
        if not item or item.get("error") is True:
            continue
        fee = item.get("fee")
        if isinstance(fee, (int, float)):
            fees.append(float(fee))

    if not fees:
        return None

    return min(fees)

# ─────────── Endpoints ───────────

@router.get("/places/autocomplete", response_model=AutocompleteLiteResponse)
async def places_autocomplete(
    input: str = Query(..., min_length=1, description="Texto parcial de dirección"),
    session_token: Optional[str] = Query(None, description="Token de sesión para mejor facturación/resultados"),
    language: str = Query("es", description="Idioma de resultados"),
    countries: Optional[str] = Query(None, description="Códigos ISO-3166-1 alpha-2 separados por | o , (p.ej. 'us|pr')"),
    limit: int = Query(5, ge=1, le=10, description="Máximo de sugerencias"),
):
    """
    Devuelve SOLO las predicciones de Google (sin nearest ni costos).
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

        if status == "ZERO_RESULTS":
            return AutocompleteLiteResponse(predictions=[], session_token=stoken)

        if status != "OK":
            msg = data.get("error_message") or status or "Error en Autocomplete"
            raise HTTPException(status_code=400, detail=msg)

        raw_preds = data.get("predictions", [])[:limit]
        items: List[AutocompleteLiteItem] = [
            AutocompleteLiteItem(
                description=p.get("description", ""),
                place_id=p.get("place_id", ""),
                types=p.get("types", []) or [],
            )
            for p in raw_preds
        ]

    return AutocompleteLiteResponse(predictions=items, session_token=stoken)

@router.get("/places/coverage-details", response_model=CoverageDetailsResponse)
async def coverage_details(
    place_id: str = Query(..., description="Place ID seleccionado por el usuario"),
    session_token: Optional[str] = Query(None, description="Token de sesión usado en Autocomplete"),
    coverage_radius_miles: float = Query(1000.0, gt=0, description="Radio de cobertura en millas"),
    language: str = Query("es", description="Idioma para Distance Matrix"),
):
    """
    Dado un place_id, resuelve su lat/lng, calcula la sede más cercana, la distancia por carretera,
    pide la tarifa a Shipday (Availability) y devuelve TODO con el MISMO SHAPE que Colombia:
    - delivery_cost_cop (int)
    - distance_km (float)
    - dropoff.address (sin extras)
    """
    async with httpx.AsyncClient() as client:
        # 1) Resolver coordenadas + address_components del destino
        dlat, dlng, formatted, comps = await places_details_with_components(client, place_id, session_token)

        # Construir dropoff.address desde components (sin raw_address)
        address, _raw_address = _build_address_from_components(comps)
        dropoff = Dropoff(address=address)

        # 2) Calcular nearest por Haversine
        near = nearest_site_for(dlat, dlng, radius_miles=coverage_radius_miles)

        if not near.in_coverage:
            # fuera de cobertura → devolvemos error pero incluimos dropoff
            return CoverageDetailsResponse(
                place_id=place_id,
                formatted_address=formatted,
                lat=dlat,
                lng=dlng,
                nearest=None,
                delivery_cost_cop=None,
                distance_km=None,
                dropoff=dropoff,
                error=make_out_of_coverage_error_by_city(address.city),
            )

        # 3) Dentro de cobertura → distancia por carretera y costo
        s = near.site["location"]

        # Distance Matrix (millas de conducción) -> km
        try:
            driving_miles, _ = await driving_distance_miles(
                client,
                s["lat"], s["long"],
                dlat, dlng,
                language=language
            )
            near.driving_distance_miles = round(driving_miles, 2)
            distance_km = driving_miles * 1.609344
        except Exception:
            # Fallback: Haversine en km
            distance_km = near.distance_miles * 1.609344

        distance_km_report = round(distance_km, DISTANCE_REPORT_DECIMALS)

        # Shipday availability (valor / fee en USD) -> usamos mismo campo delivery_cost_cop
        fee = await shipday_quote_fee(
            client,
            pickup_lat=s["lat"], pickup_lng=s["long"],
            drop_lat=dlat, drop_lng=dlng
        )

        if isinstance(fee, (int, float)):
            # Mantener nombre del campo: delivery_cost_cop (pero es USD)
            cost_int = int(math.ceil(float(fee)))
        else:
            # Fallback por millas (USD), mismo campo
            d_miles = near.driving_distance_miles or (distance_km / 1.609344)
            cost_int = int(math.ceil(max(round(d_miles * DELIVERY_RATE_USD_PER_MILE, 2), DELIVERY_RATE_USD_PER_MILE)))

    return CoverageDetailsResponse(
        place_id=place_id,
        formatted_address=formatted,
        lat=dlat,
        lng=dlng,
        nearest=near,
        delivery_cost_cop=cost_int,   # <- MISMO CAMPO
        distance_km=distance_km_report,
        dropoff=dropoff,
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
    # Resolver origen/destino
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

    # Calcular distancia según método
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
