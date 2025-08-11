#!/bin/sh
set -eu
(
  TARGET="/etc/nginx/conf.d/reverse-proxy.https.conf"
  DISABLED="/etc/nginx/conf.d/reverse-proxy.https.conf.disabled"
  FIRST_DOMAIN="api.stripe.salchimonster.com"
  # Si no hay dominios, no hacemos nada
  if [ -n "$FIRST_DOMAIN" ]; then
    # Espera a que exista el primer cert (emisión inicial)
    while [ ! -f "/etc/letsencrypt/live/$FIRST_DOMAIN/fullchain.pem" ]; do
      sleep 5
    done
    if [ -f "$DISABLED" ]; then
      cp "$DISABLED" "$TARGET"
    fi
    nginx -t && nginx -s reload || true
    # Recargas periódicas para tomar renovaciones
    while :; do
      sleep 21600  # 6h
      nginx -s reload || true
    done
  fi
) &
