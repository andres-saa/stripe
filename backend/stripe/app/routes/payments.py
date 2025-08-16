# payments.py
from fastapi import APIRouter, HTTPException, Request, Depends, Path, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dataclasses import dataclass
import os
import stripe
import logging
import httpx

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────
# Modelos
# ─────────────────────────────────────────
class CreatePaymentIntent(BaseModel):
    amount: int
    currency: str = "usd"
    metadata: dict | None = None

# ─────────────────────────────────────────
# Multi-tenant: credenciales por merchant
# ─────────────────────────────────────────
@dataclass
class MerchantCreds:
    stripe_secret_key: str
    webhook_secret: Optional[str]  # para /webhook/{merchant}
    salchi_secret: Optional[str]
    salchi_backend_base: str
    publishable_key: Optional[str]  # 👈 NUEVO

DEFAULT_SALCHI_BACKEND_BASE = os.getenv("SALCHI_BACKEND_BASE", "https://backend.salchimonster.com")

def _normalize_tenant(s: str | None) -> str:
    return (s or "default").strip().lower()

def _load_merchants_from_env() -> Dict[str, MerchantCreds]:
    """
    Carga credenciales por sufijo: VAR__TENANT.
    Ej:
      STRIPE_SECRET_KEY__US, STRIPE_WEBHOOK_SECRET__US, SALCHI_SECRET__US,
      SALCHI_BACKEND_BASE__US, STRIPE_PUBLISHABLE_KEY__US
    Crea también el tenant 'default' desde las variables sin sufijo.
    """
    buckets: Dict[str, Dict[str, str]] = {}

    # 1) Variables con sufijo __TENANT
    for k, v in os.environ.items():
        if "__" in k:
            base, tenant = k.split("__", 1)
            t = _normalize_tenant(tenant)
            buckets.setdefault(t, {})[base] = v

    # 2) Variables "default" (sin sufijo)
    buckets.setdefault("default", {})
    default_keys = [
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "WEBHOOK_SECRET",
        "SALCHI_SECRET",
        "EPAYCO_SALCHI_SECRET_KEY",
        "SALCHI_BACKEND_BASE",
        "STRIPE_PUBLISHABLE_KEY",  # 👈 NUEVO
    ]
    for base_key in default_keys:
        val = os.getenv(base_key)
        if val:
            buckets["default"][base_key] = val  # type: ignore

    # 3) Construir MerchantCreds
    result: Dict[str, MerchantCreds] = {}
    for tenant, vals in buckets.items():
        ssk = vals.get("STRIPE_SECRET_KEY")
        whs = vals.get("STRIPE_WEBHOOK_SECRET") or vals.get("WEBHOOK_SECRET")
        salchi = vals.get("SALCHI_SECRET") or vals.get("EPAYCO_SALCHI_SECRET_KEY")
        base = vals.get("SALCHI_BACKEND_BASE", DEFAULT_SALCHI_BACKEND_BASE)
        pk = vals.get("STRIPE_PUBLISHABLE_KEY")  # 👈 NUEVO

        result[tenant] = MerchantCreds(
            stripe_secret_key=ssk or "",  # validaremos al usar
            webhook_secret=whs,
            salchi_secret=salchi,
            salchi_backend_base=base,
            publishable_key=pk,           # 👈 NUEVO
        )
    return result

MERCHANTS: Dict[str, MerchantCreds] = _load_merchants_from_env()

def get_creds_or_400(merchant: str) -> MerchantCreds:
    t = _normalize_tenant(merchant)
    creds = MERCHANTS.get(t)
    if not creds:
        raise HTTPException(status_code=400, detail=f"Merchant desconocido: {t}")
    return creds

def resolve_merchant(
    request: Request,
    merchant_q: Optional[str] = Query(default=None, alias="merchant")
) -> str:
    """
    Prioridad:
    1) Path param (cuando aplique)
    2) Header: X-Merchant o X-Tenant
    3) Query param: ?merchant=
    4) 'default'
    """
    # Headers
    h = request.headers.get("x-merchant") or request.headers.get("x-tenant")
    if h and h.strip():
        return _normalize_tenant(h)
    # Query
    if merchant_q and merchant_q.strip():
        return _normalize_tenant(merchant_q)
    # Default
    return "default"

def server_mode_from_secret(sk: str | None) -> str:
    if not sk:
        return "unknown"
    return "live" if sk.startswith("sk_live_") else "test"

# ─────────────────────────────────────────
# Helpers (DB / dominio)
# ─────────────────────────────────────────
async def mark_order_paid(order_id: str, pi_id: str, merchant: str):
    """
    Llama a: {SALCHI_BACKEND_BASE}/pay-order/{order_id}/{ref}/{secret} (por merchant)
    """
    creds = get_creds_or_400(merchant)
    if not creds.salchi_secret:
        raise RuntimeError(f"[{merchant}] Falta SALCHI_SECRET en entorno")

    url = f"{creds.salchi_backend_base}/pay-order/{order_id}/{pi_id}/{creds.salchi_secret}"
    logger.info(f"[{merchant}][Stripe] Hitting pay-order endpoint: {url}")

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(url)

    if resp.status_code // 100 != 2:
        body = (resp.text or "")[:500]
        raise RuntimeError(f"[{merchant}] pay-order devolvió {resp.status_code}: {body}")

    logger.info(f"[{merchant}][Stripe] pay-order OK ({resp.status_code}): {(resp.text or '')[:200]}")

def mark_order_processing(order_id: str, pi_id: str, merchant: str) -> None:
    logger.info(f"[{merchant}][Stripe] Mark PROCESSING order={order_id} pi={pi_id}")

def mark_order_failed(order_id: str, pi_id: str | None, reason: str, merchant: str) -> None:
    logger.warning(f"[{merchant}][Stripe] Mark FAILED order={order_id} pi={pi_id} reason={reason}")

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@router.get("/stripe/config")
def stripe_config(merchant: str = Depends(resolve_merchant), request: Request = None):
    """
    Devuelve info de la cuenta según el merchant elegido por header/query.
    Header recomendado: X-Merchant: us|co|...
    """
    creds = get_creds_or_400(merchant)
    try:
        if not creds.stripe_secret_key:
            raise HTTPException(status_code=400, detail=f"[{merchant}] Falta STRIPE_SECRET_KEY")

        acct = stripe.Account.retrieve(api_key=creds.stripe_secret_key)
        return {
            "merchant": merchant,
            "server_mode": server_mode_from_secret(creds.stripe_secret_key),
            "account_id": acct.get("id"),
            "livemode": acct.get("charges_enabled", False) and acct.get("details_submitted", False),
            "publishableKey": creds.publishable_key,  # 👈 NUEVO (útil para frontend)
        }
    except Exception as e:
        logger.exception(f"[{merchant}] Error retrieving Stripe account info")
        return {
            "merchant": merchant,
            "server_mode": server_mode_from_secret(creds.stripe_secret_key),
            "error": str(e),
            "publishableKey": creds.publishable_key,  # 👈 lo devolvemos aunque falle Stripe
        }

@router.post("/create-payment-intent")
def create_payment_intent(
    body: CreatePaymentIntent,
    request: Request,
    merchant: str = Depends(resolve_merchant),
):
    """
    Crea el PaymentIntent con la API key del merchant.
    También agrega merchant al metadata para trazabilidad (no para verificación).
    Devuelve además la publishableKey correspondiente al merchant.
    """
    creds = get_creds_or_400(merchant)
    try:
        if body.amount <= 0:
            raise HTTPException(status_code=400, detail="Monto inválido")

        if not creds.stripe_secret_key:
            raise HTTPException(status_code=400, detail=f"[{merchant}] Falta STRIPE_SECRET_KEY")

        if not creds.publishable_key:
            logger.warning(f"[{merchant}] STRIPE_PUBLISHABLE_KEY no configurada")

        currency = body.currency.lower()
        metadata = dict(body.metadata or {})
        # Asegura que merchant viaje en el intent para auditoría
        metadata.setdefault("merchant", merchant)

        order_id = str(metadata.get("order_id") or "")
        request_opts: Dict[str, Any] = {}
        if order_id:
            # Incluye merchant en idempotency_key para no cruzar tenants
            request_opts["idempotency_key"] = f"pi:order:{merchant}:{order_id}:{currency}:{body.amount}"

        intent = stripe.PaymentIntent.create(
            amount=body.amount,
            currency=currency,
            automatic_payment_methods={"enabled": True},
            metadata=metadata,
            api_key=creds.stripe_secret_key,        # 👈 clave específica
            **request_opts,                          # (request option idempotency_key)
        )

        return {
            "merchant": merchant,
            "clientSecret": intent.client_secret,
            "intentId": intent.id,
            "livemode": bool(getattr(intent, "livemode", False)),
            "publishableKey": creds.publishable_key,  # 👈 NUEVO
        }

    except stripe.error.StripeError as e:
        logger.exception(f"[{merchant}] Stripe error creating PaymentIntent")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{merchant}] Unexpected error creating PaymentIntent")
        raise HTTPException(status_code=500, detail="Error creando el PaymentIntent")

@router.post("/webhook/{merchant}")
async def stripe_webhook(
    request: Request,
    merchant: str = Path(..., description="Identificador del comercio/tenant"),
):
    """
    Webhook por-merchant: verifica la firma con el secret del merchant sin confiar en el payload.
    Configura en Stripe el endpoint apuntando a /webhook/{merchant} y usa el secret de ese endpoint
    en la env var STRIPE_WEBHOOK_SECRET__{MERCHANT} (o WEBHOOK_SECRET__{MERCHANT}).
    """
    creds = get_creds_or_400(merchant)
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not creds.webhook_secret:
        logger.error(f"[{merchant}] WEBHOOK_SECRET no configurado")
        raise HTTPException(status_code=500, detail=f"[{merchant}] Webhook no configurado")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=creds.webhook_secret
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Firma inválida")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    event_type = event.get("type")
    data_obj = event.get("data", {}).get("object", {})

    try:
        if event_type == "payment_intent.succeeded":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata") or {}
            order_id = str(metadata.get("order_id") or "")
            if order_id and pi_id:
                await mark_order_paid(order_id, pi_id, merchant)

        elif event_type == "payment_intent.processing":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata") or {}
            order_id = str(metadata.get("order_id") or "")
            if order_id and pi_id:
                mark_order_processing(order_id, pi_id, merchant)

        elif event_type == "payment_intent.payment_failed":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata") or {}
            order_id = str(metadata.get("order_id") or "")
            last_payment_error = (data_obj.get("last_payment_error") or {}).get("message")
            if order_id:
                mark_order_failed(order_id, pi_id, last_payment_error or "payment_failed", merchant)

        elif event_type == "checkout.session.completed":
            # opcional
            pass

    except Exception as e:
        logger.exception(f"[{merchant}] Error manejando webhook")
        # 400 => Stripe reintenta
        raise HTTPException(status_code=400, detail=str(e))

    return {"received": True, "merchant": merchant}

@router.get("/healthz")
def healthz():
    # Devuelve listado de merchants cargados y si tienen publishable key
    return {
        "status": "ok",
        "merchants": [
            {
                "merchant": m,
                "server_mode": server_mode_from_secret(c.stripe_secret_key),
                "has_publishable_key": bool(c.publishable_key),
                "has_webhook_secret": bool(c.webhook_secret),
            }
            for m, c in MERCHANTS.items()
        ],
    }
