"""Read Pubmed Papers."""

from typing import List, Optional

from defusedxml import ElementTree as safe_xml
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


