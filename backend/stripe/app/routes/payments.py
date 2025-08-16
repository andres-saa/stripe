# payments.py
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Dict, Optional, Any
import os
import stripe
import logging
import httpx
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────
# Config multi-tenant (por sitio)
# ─────────────────────────────────────────
@dataclass
class SiteConfig:
    site_id: str
    stripe_secret_key: str
    stripe_webhook_secret: str
    salchi_secret: str
    salchi_backend_base: str

SITES: Dict[str, SiteConfig] = {}

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default

def load_sites_from_env() -> Dict[str, SiteConfig]:
    sites = {}
    ids = [s.strip() for s in (_env("STRIPE_SITES", "") or "").split(",") if s.strip()]
    # Defaults globales (opcionales)
    default_stripe_secret = _env("STRIPE_SECRET_KEY")
    default_webhook_secret = _env("STRIPE_WEBHOOK_SECRET")
    default_salchi_secret = _env("SALCHI_SECRET") or _env("EPAYCO_SALCHI_SECRET_KEY")
    default_salchi_base = _env("SALCHI_BACKEND_BASE", "https://backend.salchimonster.com")

    for sid in ids:
        cfg = SiteConfig(
            site_id=sid,
            stripe_secret_key=_env(f"STRIPE_{sid}_SECRET_KEY", default_stripe_secret) or "",
            stripe_webhook_secret=_env(f"STRIPE_{sid}_WEBHOOK_SECRET", default_webhook_secret) or "",
            salchi_secret=_env(f"SALCHI_{sid}_SECRET", default_salchi_secret) or "",
            salchi_backend_base=_env(f"SALCHI_{sid}_BACKEND_BASE", default_salchi_base) or "",
        )
        sites[sid] = cfg

    return sites

SITES = load_sites_from_env()

def get_site_config(site_id: str) -> SiteConfig:
    site_id = str(site_id)
    cfg = SITES.get(site_id)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"site_id desconocido o no configurado: {site_id}")
    if not cfg.stripe_secret_key:
        raise HTTPException(status_code=500, detail=f"Stripe SECRET KEY ausente para site_id={site_id}")
    if not cfg.stripe_webhook_secret:
        logger.warning(f"Webhook secret no configurado para site_id={site_id} (no afectará /create-payment-intent, pero sí /webhook)")
    return cfg

def server_mode_from_secret(sk: Optional[str]) -> str:
    if not sk:
        return "unknown"
    return "live" if sk.startswith("sk_live_") else "test"

# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────
class CreatePaymentIntent(BaseModel):
    site_id: str = Field(..., description="Identificador de la sede/cuenta Stripe")
    amount: int
    currency: str = "usd"
    metadata: Optional[dict] = None

# ─────────────────────────────────────────
# Helpers (DB / dominio)
# ─────────────────────────────────────────
async def mark_order_paid(order_id: str, pi_id: str, cfg: SiteConfig):
    """
    Llama a: {SALCHI_BACKEND_BASE}/pay-order/{order_id}/{ref}/{secret}
    - ref: PaymentIntent.id (pi_...)
    - secret: salchi_secret del sitio
    """
    if not cfg.salchi_secret:
        raise RuntimeError(f"Falta SALCHI_SECRET para site_id={cfg.site_id}")

    url = f"{cfg.salchi_backend_base}/pay-order/{order_id}/{pi_id}/{cfg.salchi_secret}"
    logger.info(f"[Stripe][site={cfg.site_id}] Hitting pay-order: {url}")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url)

    if resp.status_code // 100 != 2:
        body = (resp.text or "")[:500]
        raise RuntimeError(f"pay-order devolvió {resp.status_code}: {body}")

    logger.info(f"[Stripe][site={cfg.site_id}] pay-order OK ({resp.status_code}): {(resp.text or '')[:200]}")

def mark_order_processing(order_id: str, pi_id: str, cfg: SiteConfig) -> None:
    logger.info(f"[Stripe][site={cfg.site_id}] Mark PROCESSING order={order_id} pi={pi_id}")

def mark_order_failed(order_id: str, pi_id: Optional[str], reason: str, cfg: SiteConfig) -> None:
    logger.warning(f"[Stripe][site={cfg.site_id}] Mark FAILED order={order_id} pi={pi_id} reason={reason}")

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@router.get("/sites")
def list_sites():
    """Solo para diagnóstico: lista los site_id configurados (sin secretos)."""
    return {
        "sites": [
            {
                "site_id": sid,
                "server_mode": server_mode_from_secret(cfg.stripe_secret_key),
                "salchi_backend_base": cfg.salchi_backend_base,
                "has_webhook_secret": bool(cfg.stripe_webhook_secret),
            }
            for sid, cfg in SITES.items()
        ]
    }

@router.get("/stripe/config")
def stripe_config(site_id: str = Query(..., description="site_id para seleccionar la cuenta Stripe")):
    cfg = get_site_config(site_id)
    try:
        acct = stripe.Account.retrieve(api_key=cfg.stripe_secret_key)
        return {
            "site_id": site_id,
            "server_mode": server_mode_from_secret(cfg.stripe_secret_key),
            "account_id": acct.get("id"),
            "livemode": acct.get("charges_enabled", False) and acct.get("details_submitted", False),
        }
    except Exception as e:
        logger.exception("[Stripe][config] Error retrieving Stripe account info")
        return {
            "site_id": site_id,
            "server_mode": server_mode_from_secret(cfg.stripe_secret_key),
            "error": str(e),
        }

@router.post("/create-payment-intent")
def create_payment_intent(body: CreatePaymentIntent):
    cfg = get_site_config(body.site_id)

    try:
        if body.amount <= 0:
            raise HTTPException(status_code=400, detail="Monto inválido")

        currency = body.currency.lower()
        metadata = dict(body.metadata or {})
        # Aseguramos que el site_id viaje en metadata (útil en el webhook)
        metadata.setdefault("site_id", str(body.site_id))
        order_id = str(metadata.get("order_id") or "")

        idempotency_key = None
        if order_id:
            idempotency_key = f"pi:order:{order_id}:{body.site_id}:{currency}:{body.amount}"

        intent = stripe.PaymentIntent.create(
            amount=body.amount,
            currency=currency,
            automatic_payment_methods={"enabled": True},
            metadata=metadata,
            api_key=cfg.stripe_secret_key,
            idempotency_key=idempotency_key,
        )

        return {
            "site_id": body.site_id,
            "clientSecret": intent.client_secret,
            "intentId": intent.id,
            "livemode": bool(getattr(intent, "livemode", False)),
        }

    except stripe.error.StripeError as e:
        logger.exception(f"[Stripe][site={cfg.site_id}] Error creating PaymentIntent")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[Stripe][site={cfg.site_id}] Unexpected error creating PaymentIntent")
        raise HTTPException(status_code=500, detail="Error creando el PaymentIntent")

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Maneja webhooks desde múltiples cuentas Stripe:
    - Intenta validar la firma con cada STRIPE_{site}_WEBHOOK_SECRET configurado.
    - Si valida con uno, usamos ese `site` por defecto.
    - Luego, si la data trae metadata.site_id, lo tomamos como autoridad.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not SITES:
        logger.error("No hay sitios configurados (STRIPE_SITES vacío).")
        raise HTTPException(status_code=500, detail="Webhook no configurado (sin sitios)")

    verified_event = None
    verified_site: Optional[SiteConfig] = None
    last_sig_error: Optional[str] = None

    # 1) Validación contra todos los secrets conocidos
    for cfg in SITES.values():
        if not cfg.stripe_webhook_secret:
            continue
        try:
            evt = stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=cfg.stripe_webhook_secret)
            verified_event = evt
            verified_site = cfg
            break
        except stripe.error.SignatureVerificationError as e:
            # Guardamos el último error para logging
            last_sig_error = str(e)
            continue
        except Exception as e:
            last_sig_error = str(e)
            continue

    if not verified_event:
        logger.error(f"[Stripe][webhook] Firma inválida para todos los sitios. Último error: {last_sig_error}")
        raise HTTPException(status_code=400, detail="Firma inválida")

    # 2) Parse del evento
    #    NOTA: verified_event es un dict-like; accedemos con get
    event_type = verified_event.get("type")
    data_obj = verified_event.get("data", {}).get("object", {}) or {}

    # Si metadata trae site_id, lo usamos. Si no, usamos el del secret que validó.
    meta = data_obj.get("metadata") or {}
    site_id_from_meta = str(meta.get("site_id") or "").strip()
    if site_id_from_meta and site_id_from_meta in SITES:
        site_cfg = SITES[site_id_from_meta]
    else:
        site_cfg = verified_site  # fallback al sitio cuyo secret validó

    if not site_cfg:
        # Si llegamos acá, no pudimos determinar sitio (muy raro)
        logger.error("[Stripe][webhook] No fue posible determinar site_id del evento")
        raise HTTPException(status_code=400, detail="No fue posible determinar site_id del evento")

    try:
        if event_type == "payment_intent.succeeded":
            pi_id = data_obj.get("id")
            metadata = meta
            order_id = str(metadata.get("order_id") or "")
            if order_id and pi_id:
                await mark_order_paid(order_id, pi_id, site_cfg)

        elif event_type == "payment_intent.processing":
            pi_id = data_obj.get("id")
            metadata = meta
            order_id = str(metadata.get("order_id") or "")
            if order_id and pi_id:
                mark_order_processing(order_id, pi_id, site_cfg)

        elif event_type == "payment_intent.payment_failed":
            pi_id = data_obj.get("id")
            metadata = meta
            order_id = str(metadata.get("order_id") or "")
            last_payment_error = (data_obj.get("last_payment_error") or {}).get("message")
            if order_id:
                mark_order_failed(order_id, pi_id, last_payment_error or "payment_failed", site_cfg)

        elif event_type == "checkout.session.completed":
            # Si más adelante usas Checkout, aquí puedes manejar `session` y su metadata.
            pass

    except Exception as e:
        logger.exception(f"[Stripe][site={site_cfg.site_id}] Error manejando webhook")
        # 400 => Stripe reintenta
        raise HTTPException(status_code=400, detail=str(e))

    return {"received": True, "site_id": site_cfg.site_id}

@router.get("/healthz")
def healthz():
    return {"status": "ok", "sites_loaded": list(SITES.keys())}
