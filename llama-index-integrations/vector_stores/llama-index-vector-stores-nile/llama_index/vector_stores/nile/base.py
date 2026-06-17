# Standard library imports
import enum
import json
import logging
import uuid
from typing import Any, List, Tuple

# Third-party imports
import psycopg

# Local application/library specific imports
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.constants import DEFAULT_EMBEDDING_DIM
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from psycopg import sql

_logger = logging.getLogger(__name__)


class IndexType(enum.Enum):
    """Supported Index types. These are just used by a helper function to create indices."""

    PGVECTOR_IVFFLAT = 1
    PGVECTOR_HNSW = 2


class NileVectorStore(BasePydanticVectorStore):
    """Nile (Multi-tenant Postgres) Vector Store.

    Examples:
        `pip install llama-index-vector-stores-nile`

        ```python
        from llama_index.vector_stores.nile import NileVectorStore

        # Create NileVectorStore instance
        vector_store = NileVectorStore.from_params(
            service_url="postgresql://user:password@us-west-2.db.thenile.dev:5432/niledb",
            table_name="test_table",
            tenant_aware=True,
            num_dimensions=1536
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False

    service_url: str
    table_name: str
    num_dimensions: int
    tenant_aware: bool

    _sync_conn: Any = PrivateAttr()
    _async_conn: Any = PrivateAttr()

    def _create_clients(self) -> None:
        self._sync_conn = psycopg.connect(self.service_url)
        self._async_conn = psycopg.connect(self.service_url)

    def _create_tables(self) -> None:
        _logger.info(
            f"Creating tables for {self.table_name} with {self.num_dimensions} dimensions"
        )
        with self._sync_conn.cursor() as cursor:
            if self.tenant_aware:
                query = sql.SQL(
                    """
                        CREATE TABLE IF NOT EXISTS {table_name}
                        (id UUID DEFAULT (gen_random_uuid()), tenant_id UUID, embedding VECTOR({num_dimensions}), content TEXT, metadata JSONB)
                    """
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    num_dimensions=sql.Literal(self.num_dimensions),
                )
                cursor.execute(query)
            else:
                query = sql.SQL(
                    """
                        CREATE TABLE IF NOT EXISTS {table_name}
                        (id UUID DEFAULT (gen_random_uuid()), embedding VECTOR({num_dimensions}), content TEXT, metadata JSONB)
                    """
                ).format(
                    table_name=sql.Identifier(self.table_name),
                    num_dimensions=sql.Literal(self.num_dimensions),
                )
                cursor.execute(query)
        self._sync_conn.commit()

    def __init__(
        self,
        service_url: str,
        table_name: str,
        tenant_aware: bool = False,
        num_dimensions: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        super().__init__(
            service_url=service_url,
            table_name=table_name,
            num_dimensions=num_dimensions,
            tenant_aware=tenant_aware,
        )

        self._create_clients()
        self._create_tables()

    @classmethod
    def class_name(cls) -> str:
        return "NileVectorStore"

    @property
    def client(self) -> Any:
        return self._sync_conn

    async def close(self) -> None:
        self._sync_conn.close()
        await self._async_conn.close()

    @classmethod
    def from_params(
        cls,
        service_url: str,
        table_name: str,
        tenant_aware: bool = False,
        num_dimensions: int = DEFAULT_EMBEDDING_DIM,
    ) -> "NileVectorStore":
        return cls(
            service_url=service_url,
            table_name=table_name,
            tenant_aware=tenant_aware,
            num_dimensions=num_dimensions,
        )

    # We extract tenant_id from the node metadata.
    def _node_to_row(self, node: BaseNode) -> Any:
        metadata = node_to_metadata_dict(
            node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )
        tenant_id = node.metadata.get("tenant_id", None)
        return [
            tenant_id,
            metadata,
            node.get_content(metadata_mode=MetadataMode.NONE),
            node.embedding,
        ]

    def _insert_row(self, cursor: Any, row: Any) -> str:
        _logger.debug(f"Inserting row into {self.table_name} with tenant_id {row[0]}")
        if self.tenant_aware:
            if row[0] is None:
                # Nile would fail the insert itself, but this saves the DB call and easier to test
                raise ValueError("tenant_id cannot be None if tenant_aware is True")
            query = sql.SQL(
                """
                           INSERT INTO {} (tenant_id, metadata, content, embedding) VALUES (%(tenant_id)s, %(metadata)s, %(content)s, %(embedding)s) returning id
                       """
            ).format(sql.Identifier(self.table_name))
            cursor.execute(
                query,
                {
                    "tenant_id": row[0],
                    "metadata": json.dumps(row[1]),
                    "content": row[2],
                    "embedding": row[3],
                },
            )
        else:
            query = sql.SQL(
                """
                           INSERT INTO {} (metadata, content, embedding) VALUES (%(metadata)s, %(content)s, %(embedding)s) returning id
                       """
            ).format(sql.Identifier(self.table_name))
            cursor.execute(
                query,
                {
                    "metadata": json.dumps(row[0]),
                    "content": row[1],
                    "embedding": row[2],
                },
            )
        id = cursor.fetchone()[0]
        self._sync_conn.commit()
        return id

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        rows_to_insert = [self._node_to_row(node) for node in nodes]
        ids = []
        with self._sync_conn.cursor() as cursor:
            for row in rows_to_insert:
                # this will throw an error if tenant_id is None and tenant_aware is True, which is what we want
                ids.append(
                    self._insert_row(cursor, row)
                )  # commit is called in _insert_row
        return ids

    async def async_add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        rows_to_insert = [self._node_to_row(node) for node in nodes]
        ids = []
        async with self._async_conn.cursor() as cursor:
            for row in rows_to_insert:
                ids.append(self._insert_row(cursor, row))
            await self._async_conn.commit()
        return ids

    def _set_tenant_context(self, cursor: Any, tenant_id: Any) -> None:
        if self.tenant_aware:
            cursor.execute(
                sql.SQL(""" set local nile.tenant_id = {} """).format(
                    sql.Literal(tenant_id)
                )
            )

    def _to_postgres_operator(self, operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return "="
        elif operator == FilterOperator.GT:
            return ">"
        elif operator == FilterOperator.LT:
            return "<"
        elif operator == FilterOperator.NE:
            return "!="
        elif operator == FilterOperator.GTE:
            return ">="
        elif operator == FilterOperator.LTE:
            return "<="
        elif operator == FilterOperator.IN:
            return "IN"
        elif operator == FilterOperator.NIN:
            return "NOT IN"
        elif operator == FilterOperator.CONTAINS:
            return "@>"
        elif operator == FilterOperator.TEXT_MATCH:
            return "LIKE"
        elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
            return "ILIKE"
        else:
            _logger.warning(f"Unknown operator: {operator}, fallback to '='")
            return "="
