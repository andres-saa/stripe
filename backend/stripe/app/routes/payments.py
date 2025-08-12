# payments.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import os
import stripe
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Claves
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────
class CreatePaymentIntent(BaseModel):
    # Monto en UNIDADES MÍNIMAS (minor units). Ej:
    # 35.000 COP => 350000 ; 25 USD => 2500
    amount: int
    currency: str = "usd"       # 'usd' o 'cop', etc.
    # Pasa order_id para idempotencia y conciliación en el webhook
    metadata: dict | None = None  # ej: {"order_id": "SM-12345", "site_id": 7, "user_id": "u1"}

# ─────────────────────────────────────────
# Helpers (DB / dominio)
# ─────────────────────────────────────────
def mark_order_paid(order_id: str, pi_id: str, amount: int, currency: str) -> None:
    logger.info(f"[Stripe] Mark PAID order={order_id} pi={pi_id} amount={amount} {currency}")

def mark_order_processing(order_id: str, pi_id: str) -> None:
    logger.info(f"[Stripe] Mark PROCESSING order={order_id} pi={pi_id}")

def mark_order_failed(order_id: str, pi_id: str | None, reason: str) -> None:
    logger.warning(f"[Stripe] Mark FAILED order={order_id} pi={pi_id} reason={reason}")

def server_mode_from_secret(sk: str | None) -> str:
    if not sk:
        return "unknown"
    return "live" if sk.startswith("sk_live_") else "test"

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@router.get("/stripe/config")
def stripe_config():
    """
    Útil para depurar: confirma el modo del servidor y la cuenta de Stripe
    contra la que estás creando PaymentIntents.
    """
    try:
        acct = stripe.Account.retrieve()
        return {
            "server_mode": server_mode_from_secret(os.getenv("STRIPE_SECRET_KEY")),
            "account_id": acct.get("id"),
            "livemode": acct.get("charges_enabled", False) and acct.get("details_submitted", False),
            # Nota: 'livemode' aquí no es 1:1, lo importante es 'server_mode' y 'account_id'
        }
    except Exception as e:
        logger.exception("Error retrieving Stripe account info")
        # No es crítico para el flujo de cobro
        return {"server_mode": server_mode_from_secret(os.getenv("STRIPE_SECRET_KEY")), "error": str(e)}

@router.post("/create-payment-intent")
def create_payment_intent(body: CreatePaymentIntent):
    """
    Crea un PaymentIntent para usar con Payment Element (frontend).
    El front ya envía 'amount' en minor units (centavos).
    """
    try:
        if body.amount <= 0:
            raise HTTPException(status_code=400, detail="Monto inválido")

        currency = body.currency.lower()
        metadata = body.metadata or {}
        order_id = str(metadata.get("order_id") or "")

        # Idempotencia por order_id (recomendado si reintentas)
        request_opts = {}
        if order_id:
            request_opts["idempotency_key"] = f"pi:order:{order_id}:{currency}:{body.amount}"

        intent = stripe.PaymentIntent.create(
            amount=body.amount,
            currency=currency,
            automatic_payment_methods={"enabled": True},
            metadata=metadata,
            **request_opts,
        )

        # Devuelve datos útiles para verificar modo/cuenta en el front
        return {
            "clientSecret": intent.client_secret,
            "intentId": intent.id,
            "livemode": bool(getattr(intent, "livemode", False)),
        }

    except stripe.error.StripeError as e:
        logger.exception("Stripe error creating PaymentIntent")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error creating PaymentIntent")
        raise HTTPException(status_code=500, detail="Error creando el PaymentIntent")

@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Webhook de Stripe. Configura en el Dashboard/CLI:
      https://api.stripe.salchimonster.com/webhook
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not WEBHOOK_SECRET:
        logger.error("WEBHOOK_SECRET no configurado")
        raise HTTPException(status_code=500, detail="Webhook no configurado")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=WEBHOOK_SECRET
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
            amount = data_obj.get("amount", 0)
            currency = (data_obj.get("currency") or "").lower()
            metadata = data_obj.get("metadata") or {}
            order_id = str(metadata.get("order_id") or "")
            if order_id:
                mark_order_paid(order_id, pi_id, amount, currency)

        elif event_type == "payment_intent.processing":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata") or {}
            order_id = str(metadata.get("order_id") or "")
            if order_id:
                mark_order_processing(order_id, pi_id)

        elif event_type == "payment_intent.payment_failed":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata") or {}
            order_id = str(metadata.get("order_id") or "")
            last_payment_error = (data_obj.get("last_payment_error") or {}).get("message")
            if order_id:
                mark_order_failed(order_id, pi_id, last_payment_error or "payment_failed")

        elif event_type == "checkout.session.completed":
            # Si en el futuro usas Checkout, manéjalo acá.
            pass

    except Exception as e:
        logger.exception("Error manejando webhook")
        # 4xx/5xx hace que Stripe reintente; usa 4xx si es un error recuperable
        raise HTTPException(status_code=400, detail=str(e))

    return {"received": True}

@router.get("/healthz")
def healthz():
    return {"status": "ok"}
