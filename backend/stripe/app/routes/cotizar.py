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
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
DELIVERY_RATE_USD_PER_MILE = float(os.getenv("DELIVERY_RATE_USD_PER_MILE", "1.5"))


router = APIRouter()

# ─────────── SEDES (tus datos) ───────────
SEDES: List[Dict[str, Any]] = [
    {
        "site_id": 33,
        "site_name": "UNION CITY",
        "site_address": "2100 kerrigan ave union city nj 07087",
        "pe_site_id": 16,
        "location": {"lat": 40.76808, "long": -74.03843},
    },
    {
        "site_id": 35,
        "site_name": "FILADELPHIA",
        "site_address": "5759 Oxford ave, Philadelphia, PA 19149",
        "pe_site_id": 16,
        "location": {"lat": 40.03358, "long": -75.08501},
    },
    {
        "site_id": 36,
        "site_name": "NEWARK",
        "site_address": "26 eckert ave Newark NJ 07112",
        "pe_site_id": 16,
        "location": {"lat": 40.71335, "long": -74.20744},
    },
]




PLACES_COUNTRIES = os.getenv("PLACES_COUNTRIES", "us")  # p.ej. "us" o "us|pr"

def _components_for(countries: Optional[str] = None) -> Optional[str]:
    """
    Convierte 'us|pr' o 'usa, puerto-rico' -> 'country:us|country:pr'
    Filtra y normaliza a ISO-3166-1 alpha-2.
    """
    countries = countries or PLACES_COUNTRIES  # p.ej. 'us'
    raw = [t.strip().lower() for t in re.split(r"[,\| ]+", countries) if t.strip()]

    # Normalizaciones comunes
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

    # dedup preservando orden
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
    distance_miles: float
    distance_km: float
    method: str = "great_circle_haversine"

class NearestInfo(BaseModel):
    site: Dict[str, Any]
    distance_miles: float
    in_coverage: bool

class AutocompleteItem(BaseModel):
    description: str
    place_id: str
    types: List[str] = []
    nearest: Optional[NearestInfo] = None  # sede más cercana + bandera de cobertura
    delivery_cost_usd: Optional[float] = None  # <--- NUEVO


class CoverageError(BaseModel):
    code: str
    message_es: str
    message_en: str
    coverage_radius_miles: float

class AutocompleteResponse(BaseModel):
    predictions: List[AutocompleteItem]
    session_token: str
    error: Optional[CoverageError] = None  # <-- NUEVO: error solo si ninguna está en cobertura

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

def nearest_site_for(lat: float, lng: float, radius_miles: float = 8.0) -> NearestInfo:
    best = None
    best_dist = float("inf")
    for s in SEDES:
        slat = s["location"]["lat"]
        slng = s["location"]["long"]  # OJO: clave 'long' en tu JSON
        d = haversine_miles(lat, lng, slat, slng)
        if d < best_dist:
            best_dist = d
            best = s
    return NearestInfo(site=best, distance_miles=round(best_dist,2), in_coverage=(best_dist <= radius_miles))

def make_out_of_coverage_error(radius_miles: float) -> CoverageError:
    return CoverageError(
        code="OUT_OF_COVERAGE",
        message_es=f"Fuera de rango: no hay cobertura dentro de {radius_miles} millas de nuestras sedes.",
        message_en=f"Out of range: no coverage within {radius_miles} miles of our locations.",
        coverage_radius_miles=radius_miles,
    )

async def geocode_address(client: httpx.AsyncClient, address: str) -> Tuple[float, float, str]:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {"address": address, "key": GOOGLE_API_KEY, "components": _components_for()},  # usa PLACES_COUNTRIES ("us") por defecto}
    resp = await client.get(GEOCODE_URL, params=params, timeout=20.0)
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
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "formatted_address,geometry/location"
    }
    if session_token:
        params["sessiontoken"] = session_token
    resp = await client.get(PLACES_DETAILS_URL, params=params, timeout=20.0)
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

# ─────────── Endpoints ───────────
@router.get("/places/autocomplete", response_model=AutocompleteResponse)
async def places_autocomplete(
    input: str = Query(..., min_length=1, description="Texto parcial de dirección"),
    session_token: Optional[str] = Query(None, description="Token de sesión para mejor facturación/resultados"),
    language: str = Query("es", description="Idioma de resultados"),
    # ⚠️ 'region' NO es parámetro de Autocomplete. La dejamos por compatibilidad,
    # pero la convertimos en un 'country' más dentro de components.
    region: Optional[str] = Query(None, description="Código de país ISO-3166-1 alpha-2 (deprecated)"),
    limit: int = Query(5, ge=1, le=10, description="Máximo de sugerencias"),
    coverage_radius_miles: float = Query(8.0, gt=0, description="Radio de cobertura en millas"),
    countries: Optional[str] = Query(None, description="Códigos ISO-3166-1 alpha-2 separados por | o , (p.ej. 'us|pr')"),
):
    """
    Devuelve SIEMPRE las predicciones de Google, con `nearest`.
    Si ninguna cae en cobertura, incluye `error` (200 OK).
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="No se configuró GOOGLE_MAPS_API_KEY")

    stoken = session_token or str(uuid.uuid4())

    # Construir components de manera robusta
    comps = _components_for(countries)
    # fusionar 'region' si llega
    if region:
        reg = _components_for(region)
        comps = f"{comps}|{reg}" if (comps and reg) else (reg or comps)

    if SEDES:
        CENTER_LAT = sum(s["location"]["lat"] for s in SEDES) / len(SEDES)
        CENTER_LNG = sum(s["location"]["long"] for s in SEDES) / len(SEDES)  # ojo: clave 'long' en tu JSON
    else:
        CENTER_LAT = 0.0
        CENTER_LNG = 0.0

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
            return AutocompleteResponse(predictions=[], session_token=stoken)

        if status != "OK":
            msg = data.get("error_message") or status or "Error en Autocomplete"
            raise HTTPException(status_code=400, detail=msg)

        raw_preds = data.get("predictions", [])[:limit]
        items: List[AutocompleteItem] = [
            AutocompleteItem(
                description=p.get("description", ""),
                place_id=p.get("place_id", ""),
                types=p.get("types", []) or [],
            )
            for p in raw_preds
        ]

        # Resolver coordenadas para nearest
        async with httpx.AsyncClient() as client2:
            tasks = [places_details(client2, it.place_id, stoken) for it in items]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for it, res in zip(items, results):
            if isinstance(res, Exception):
                continue
            plat, plng, _formatted = res
            it.nearest = nearest_site_for(plat, plng, radius_miles=coverage_radius_miles)

        any_in_coverage = any(it.nearest and it.nearest.in_coverage for it in items)

        if not any_in_coverage and items:
            return AutocompleteResponse(
                predictions=items,
                session_token=stoken,
                error=make_out_of_coverage_error(coverage_radius_miles),
            )

        for it, res in zip(items, results):
            if isinstance(res, Exception):
                continue
            plat, plng, _formatted = res
            it.nearest = nearest_site_for(plat, plng, radius_miles=coverage_radius_miles)

            # Asignar costo solo si está en cobertura
            if it.nearest and it.nearest.in_coverage:
                it.delivery_cost_usd = math.ceil(
                    max(
                        round(it.nearest.distance_miles * DELIVERY_RATE_USD_PER_MILE, 2),
                        DELIVERY_RATE_USD_PER_MILE
                )
                )
            else:
                it.delivery_cost_usd = None

                

        return AutocompleteResponse(predictions=items, session_token=stoken)


@router.get("/places/details", response_model=GeocodedPoint)
async def places_details_endpoint(
    place_id: str = Query(..., description="Place ID a resolver"),
    session_token: Optional[str] = Query(None, description="Token de sesión usado en Autocomplete")
):
    async with httpx.AsyncClient() as client:
        lat, lng, formatted = await places_details(client, place_id, session_token)
    return GeocodedPoint(query=place_id, formatted_address=formatted, lat=lat, lng=lng)

@router.post("/distance", response_model=DistanceResponse)
async def compute_distance(body: DistanceRequest):
    async with httpx.AsyncClient() as client:
        # Origen
        if body.origin_place_id:
            olat, olng, ofmt = await places_details(client, body.origin_place_id, body.session_token)
            oquery = body.origin_place_id
        elif body.origin:
            olat, olng, ofmt = await geocode_address(client, body.origin)
            oquery = body.origin
        else:
            raise HTTPException(status_code=422, detail="Debes enviar origin o origin_place_id")

        # Destino
        if body.destination_place_id:
            dlat, dlng, dfmt = await places_details(client, body.destination_place_id, body.session_token)
            dquery = body.destination_place_id
        elif body.destination:
            dlat, dlng, dfmt = await geocode_address(client, body.destination)
            dquery = body.destination
        else:
            raise HTTPException(status_code=422, detail="Debes enviar destination o destination_place_id")

    miles = haversine_miles(olat, olng, dlat, dlng)
    km = miles * 1.609344

    origin_gc = GeocodedPoint(query=oquery, formatted_address=ofmt, lat=olat, lng=olng)
    destination_gc = GeocodedPoint(query=dquery, formatted_address=dfmt, lat=dlat, lng=dlng)

    return DistanceResponse(
        origin=origin_gc,
        destination=destination_gc,
        distance_miles=round(miles, 2),
        distance_km=round(km, 2),
    )
