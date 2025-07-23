import os
import logging
import sys
from flask_talisman import Talisman
from urllib.parse import urlparse

def apply_flask_security(app):
    app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

    csp = {
        'default-src': ["'self'"],
        'media-src': ['rtsp:', 'http:', 'https:']
    }

    Talisman(
        app,
        content_security_policy=csp,
        force_https=True,
        strict_transport_security=True,
        strict_transport_security_max_age=31536000,
        strict_transport_security_include_subdomains=True,
        frame_options='DENY'
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('error.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return app

def is_valid_url(url):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https', 'rtsp']:
            return False
        if not parsed.netloc:
            return False
        if ' ' in url:
            return False
        return True
    except Exception as e:
        logging.error(f"Invalid URL check error: {e}")
        return False
