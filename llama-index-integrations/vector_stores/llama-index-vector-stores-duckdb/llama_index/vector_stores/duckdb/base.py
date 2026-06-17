"""DuckDB vector store."""

import json
import logging
import os
from typing import Any, List, Optional, cast

from fsspec.utils import Sequence
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

import duckdb

logger = logging.getLogger(__name__)
import_err_msg = "`duckdb` package not found, please run `pip install duckdb`"





