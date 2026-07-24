"""Read Pubmed Papers."""

from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class PubmedReader(BaseReader):
    """Pubmed Reader.

    Gets a search query, return a list of Documents of the top corresponding scientific papers on Pubmed.
    """

