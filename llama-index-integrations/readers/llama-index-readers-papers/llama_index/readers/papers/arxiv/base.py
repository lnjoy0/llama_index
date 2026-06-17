"""Read Arxiv Papers."""

import hashlib
import logging
import os
from typing import List, Optional, Tuple

from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


