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
# Multi-tenant por SEDE (site_id)
# ─────────────────────────────────────────
@dataclass
class MerchantCreds:
    stripe_secret_key: str
    webhook_secret: Optional[str]
    salchi_secret: Optional[str]
    salchi_backend_base: str
    publishable_key: Optional[str]

DEFAULT_SALCHI_BACKEND_BASE = os.getenv("SALCHI_BACKEND_BASE", "https://backend.salchimonster.com")

def _normalize(s: str | None) -> str:
    return (s or "").strip().lower()

def _load_merchants_from_env() -> Dict[str, MerchantCreds]:
    """
    Carga credenciales POR SEDE usando:
      STRIPE_SITES=33,35,36

    Para cada sede X:
      STRIPE_X_SECRET_KEY
      STRIPE_X_WEBHOOK_SECRET
      STRIPE_X_PUBLISHABLE_KEY      (opcional)
      SALCHI_X_SECRET               (opcional; fallback EPAYCO_SALCHI_SECRET_KEY)
      SALCHI_X_BACKEND_BASE         (opcional; fallback DEFAULT_SALCHI_BACKEND_BASE)

    ⚠️ No se crea 'default'. Si no llega site/merchant válido => 400.
    """
    result: Dict[str, MerchantCreds] = {}

    sites_raw = os.getenv("STRIPE_SITES", "")
    sites = [i.strip() for i in sites_raw.split(",") if i.strip()]

    for site in sites:
        ssk = os.getenv(f"STRIPE_{site}_SECRET_KEY", "")
        whs = os.getenv(f"STRIPE_{site}_WEBHOOK_SECRET") or os.getenv(f"WEBHOOK_{site}_SECRET")
        salchi = os.getenv(f"SALCHI_{site}_SECRET") or os.getenv("EPAYCO_SALCHI_SECRET_KEY")
        base = os.getenv(f"SALCHI_{site}_BACKEND_BASE") or DEFAULT_SALCHI_BACKEND_BASE
        pk = os.getenv(f"STRIPE_{site}_PUBLISHABLE_KEY") or os.getenv("STRIPE_PUBLISHABLE_KEY")

        result[_normalize(site)] = MerchantCreds(
            stripe_secret_key=ssk,
            webhook_secret=whs,
            salchi_secret=salchi,
            salchi_backend_base=base,
            publishable_key=pk,
        )

    return result

MERCHANTS: Dict[str, MerchantCreds] = _load_merchants_from_env()

def get_creds_or_400(merchant: str) -> MerchantCreds:
    t = _normalize(merchant)
    creds = MERCHANTS.get(t)
    if not creds:
        sitios = ", ".join(sorted(MERCHANTS.keys())) or "(vacío)"
        raise HTTPException(
            status_code=400,
            detail=f"Merchant/Site desconocido: {t}. Válidos: {sitios}"
        )
    return creds

def resolve_merchant(
    request: Request,
    merchant_q: Optional[str] = Query(default=None, alias="merchant"),
    site_q: Optional[str] = Query(default=None, alias="site"),
) -> str:
    """
    Identifica la SEDE/TENANT. Ahora es OBLIGATORIO.
    Acepta:
      Headers: X-Site | X-Site-Id | X-Merchant | X-Tenant
      Query:   ?site=   | ?merchant=
    Si no llega o no existe en MERCHANTS => 400.
    """
    # 1) Headers
    for hk in ("x-site", "x-site-id", "x-merchant", "x-tenant"):
        hv = request.headers.get(hk)
        if hv and hv.strip():
            t = _normalize(hv)
            if t not in MERCHANTS:
                sitios = ", ".join(sorted(MERCHANTS.keys())) or "(vacío)"
                raise HTTPException(
                    status_code=400,
                    detail=f"Merchant/Site desconocido: {t}. Debe enviar un site_id válido. Válidos: {sitios}"
                )
            return t

    # 2) Query
    for v in (site_q, merchant_q):
        if v and v.strip():
            t = _normalize(v)
            if t not in MERCHANTS:
                sitios = ", ".join(sorted(MERCHANTS.keys())) or "(vacío)"
                raise HTTPException(
                    status_code=400,
                    detail=f"Merchant/Site desconocido: {t}. Debe enviar un site_id válido. Válidos: {sitios}"
                )
            return t

    # 3) Nada => 400 (sin default)
    raise HTTPException(
        status_code=400,
        detail="site_id/merchant es obligatorio. Envíe X-Site|X-Site-Id|X-Merchant|X-Tenant o ?site=|?merchant=."
    )

def server_mode_from_secret(sk: str | None) -> str:
    if not sk:
        return "unknown"
    return "live" if str(sk).startswith("sk_live_") else "test"

# ─────────────────────────────────────────
# Helpers (DB / dominio)
# ─────────────────────────────────────────
async def mark_order_paid(order_id: str, pi_id: str, merchant: str):
    """
    Llama a: {SALCHI_BACKEND_BASE}/pay-order/{order_id}/{ref}/{secret} (por merchant/sede)
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
def stripe_config(merchant: str = Depends(resolve_merchant)):
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
            "publishableKey": creds.publishable_key,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{merchant}] Error retrieving Stripe account info")
        return {
            "merchant": merchant,
            "server_mode": server_mode_from_secret(creds.stripe_secret_key),
            "error": str(e),
            "publishableKey": creds.publishable_key,
        }

@router.post("/create-payment-intent")
def create_payment_intent(
    body: CreatePaymentIntent,
    request: Request,
    merchant: str = Depends(resolve_merchant),
):
    """
    Crea el PaymentIntent con la API key de la SEDE (merchant).
    Incluye el 'merchant' en metadata para auditoría.
    Devuelve también la publishableKey correspondiente a esa sede.
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
        metadata.setdefault("merchant", merchant)

        order_id = str(metadata.get("order_id") or "")
        request_opts: Dict[str, Any] = {}
        if order_id:
            # Stripe Python permite pasar idempotency_key como kwarg top-level.
            request_opts["idempotency_key"] = f"pi:order:{merchant}:{order_id}:{currency}:{body.amount}"

        intent = stripe.PaymentIntent.create(
            amount=body.amount,
            currency=currency,
            automatic_payment_methods={"enabled": True},
            metadata=metadata,
            api_key=creds.stripe_secret_key,
            **request_opts,
        )

        return {
            "merchant": merchant,
            "clientSecret": intent.client_secret,
            "intentId": intent.id,
            "livemode": bool(getattr(intent, "livemode", False)),
            "publishableKey": creds.publishable_key,
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
    merchant: str = Path(..., description="Identificador de la sede/merchant (site_id)"),
):
    """
    Webhook por sede: verifica la firma con el secret de ESA sede.
    Configura en Stripe el endpoint apuntando a /webhook/{siteId}
    y usa en el entorno STRIPE_{site}_WEBHOOK_SECRET.
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
    # Devuelve listado de merchants/sedes cargados y banderas útiles
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
