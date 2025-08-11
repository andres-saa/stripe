# payments.py
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import stripe, os

router = APIRouter()
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class CreatePaymentIntent(BaseModel):
    amount: int            # en la unidad más pequeña (p. ej., USD = centavos)
    currency: str = "usd"  # ajusta según necesidad
    metadata: dict | None = None

@router.post("/create-payment-intent")
def create_payment_intent(body: CreatePaymentIntent):
    try:
        intent = stripe.PaymentIntent.create(
            amount=body.amount,
            currency=body.currency,
            automatic_payment_methods={"enabled": True},
            metadata=body.metadata or {}
        )
        return {"clientSecret": intent.client_secret}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))





stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig_header, secret=WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Firma inválida")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Manejo de eventos
    if event["type"] == "payment_intent.succeeded":
        pi = event["data"]["object"]
        # TODO: marcar orden como pagada (usa pi["id"] y pi["metadata"])
    elif event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        # TODO: cumplir orden (session["id"], session["metadata"])

    return {"received": True}
