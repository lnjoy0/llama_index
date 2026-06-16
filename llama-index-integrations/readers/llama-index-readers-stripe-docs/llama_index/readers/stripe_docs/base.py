import urllib.request
from typing import List

from defusedxml.ElementTree import fromstring
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.web import AsyncWebPageReader

XML_SITEMAP_SCHEMA = "http://www.sitemaps.org/schemas/sitemap/0.9"
STRIPE_SITEMAP_URL = "https://stripe.com/sitemap/sitemap.xml"

DEFAULT_FILTERS = ["/docs"]


