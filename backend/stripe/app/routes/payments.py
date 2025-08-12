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
    # Conviene pasar order_id para idempotencia y poder conciliar en el webhook
    metadata: dict | None = None  # ej: {"order_id": "SM-12345", "site_id": 7, "user_id": "u1"}

# ─────────────────────────────────────────
# Helpers (DB / dominio)
# ─────────────────────────────────────────
def mark_order_paid(order_id: str, pi_id: str, amount: int, currency: str) -> None:
    """
    TODO: Implementa la actualización real en tu DB:
      - orders.status = 'paid'
      - orders.payment_provider = 'stripe'
      - orders.payment_intent_id = pi_id
      - orders.paid_amount_minor = amount
      - orders.currency = currency
      - orders.paid_at = now()
    """
    logger.info(f"[Stripe] Mark PAID order={order_id} pi={pi_id} amount={amount} {currency}")

def mark_order_processing(order_id: str, pi_id: str) -> None:
    # Opcional: si quieres llevar un estado intermedio
    logger.info(f"[Stripe] Mark PROCESSING order={order_id} pi={pi_id}")

def mark_order_failed(order_id: str, pi_id: str | None, reason: str) -> None:
    logger.warning(f"[Stripe] Mark FAILED order={order_id} pi={pi_id} reason={reason}")

# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@router.post("/create-payment-intent")
def create_payment_intent(body: CreatePaymentIntent):
    """
    Crea un PaymentIntent para usar con Payment Element (frontend).
    El front ya envía 'amount' en minor units (centavos).
    """
    try:
        if body.amount <= 0:
            raise HTTPException(status_code=400, detail="Monto inválido")

        metadata = body.metadata or {}
        order_id = str(metadata.get("order_id") or "")

        # Idempotencia por order_id (opcional pero recomendado si reintentas)
        # Si no hay order_id, no enviamos idempotency_key.
        request_opts = {}
        if order_id:
            request_opts["idempotency_key"] = f"pi:order:{order_id}:{body.currency}:{body.amount}"

        intent = stripe.PaymentIntent.create(
            amount=body.amount,
            currency=body.currency.lower(),
            automatic_payment_methods={"enabled": True},
            metadata=metadata,
            **request_opts
        )

        return {"clientSecret": intent.client_secret}

    except stripe.error.StripeError as e:
        # Errores legibles de Stripe
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
        # Opcional: aceptar sin verificar SOLO en dev (no recomendado en prod).
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
            currency = data_obj.get("currency", "").lower()
            metadata = data_obj.get("metadata", {}) or {}
            order_id = str(metadata.get("order_id") or "")
            if order_id:
                mark_order_paid(order_id, pi_id, amount, currency)

        elif event_type == "payment_intent.processing":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata", {}) or {}
            order_id = str(metadata.get("order_id") or "")
            if order_id:
                mark_order_processing(order_id, pi_id)

        elif event_type == "payment_intent.payment_failed":
            pi_id = data_obj.get("id")
            metadata = data_obj.get("metadata", {}) or {}
            order_id = str(metadata.get("order_id") or "")
            last_payment_error = (data_obj.get("last_payment_error") or {}).get("message")
            if order_id:
                mark_order_failed(order_id, pi_id, last_payment_error or "payment_failed")

        # Si en el futuro usas Checkout, puedes manejarlo aquí también:
        elif event_type == "checkout.session.completed":
            session = data_obj
            # order_id = session.get("metadata", {}).get("order_id")
            # TODO opcional: mark_order_paid(order_id, session["payment_intent"], amount, currency)

    except Exception as e:
        logger.exception("Error manejando webhook")
        # No devuelvas 500 por eventos que quieras reintentar; Stripe reintenta con 4xx/5xx
        raise HTTPException(status_code=400, detail=str(e))

    return {"received": True}


@router.get("/healthz")
def healthz():
    return {"status": "ok"}
