#!/usr/bin/env bash

# start-server.sh
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ] ; then
    (cd mldeployed; python manage.py createsuperuser --no-input)
fi
(cd mldeployed; python3 manage.py collectstatic --noinput; gunicorn mldeployed.wsgi --user www-data --bind 0.0.0.0:8010 --workers 3) &
nginx -g "daemon off;"