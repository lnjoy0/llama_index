import urllib.request
import xml.etree.ElementTree as ET
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.web import AsyncWebPageReader

XML_SITEMAP_SCHEMA = "http://www.sitemaps.org/schemas/sitemap/0.9"
STRIPE_SITEMAP_URL = "https://stripe.com/sitemap/sitemap.xml"

DEFAULT_FILTERS = ["/docs"]


