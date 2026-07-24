"""DuckDB vector store."""

import logging
import json
from typing import Any, List, Optional
import os
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)
import_err_msg = "`duckdb` package not found, please run `pip install duckdb`"


class DuckDBLocalContext:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self._conn = None
        self._home_dir = os.path.expanduser("~")





    @classmethod
    def class_name(cls) -> str:
        return "DuckDBVectorStore"

    @property
    def client(self) -> Any:
        """Return client."""
        return self._conn


    def _node_to_table_row(self, node: BaseNode) -> Any:
        return (
            node.node_id,
            node.get_content(metadata_mode=MetadataMode.NONE),
            node.get_embedding(),
            node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        )

    def _table_row_to_node(self, row: Any) -> BaseNode:
        return metadata_dict_to_node(json.loads(row[3]), row[1])


