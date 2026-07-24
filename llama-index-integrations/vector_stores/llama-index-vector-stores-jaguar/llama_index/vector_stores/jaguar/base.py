""" Jaguar Vector Store.

. A distributed vector database
. The ZeroMove feature enables instant horizontal scalability
. Multimodal: embeddings, text, images, videos, PDFs, audio, time series, and geospatial
. All-masters: allows both parallel reads and writes
. Anomaly detection capabilities: anomaly and anomamous
. RAG support: combines LLMs with proprietary and real-time data
. Shared metadata: sharing of metadata across multiple vector indexes
. Distance metrics: Euclidean, Cosine, InnerProduct, Manhatten, Chebyshev, Hamming, Jeccard, Minkowski

"""

import datetime
import json
import logging
from typing import Any, List, Optional, Tuple, Union, cast

from jaguardb_http_client.JaguarHttpClient import JaguarHttpClient

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)


class JaguarVectorStore(BasePydanticVectorStore):
    """Jaguar vector store.

    See http://www.jaguardb.com
    See http://github.com/fserv/jaguar-sdk

    Examples:
        `pip install llama-index-vector-stores-jaguar`

        ```python
        from llama_index.vector_stores.jaguar import JaguarVectorStore
        vectorstore = JaguarVectorStore(
            pod = 'vdb',
            store = 'mystore',
            vector_index = 'v',
            vector_type = 'cosine_fraction_float',
            vector_dimension = 1536,
            url='http://192.168.8.88:8080/fwww/',
        )
        ```
    """

    stores_text: bool = True

    _pod: str = PrivateAttr()
    _store: str = PrivateAttr()
    _vector_index: str = PrivateAttr()
    _vector_type: str = PrivateAttr()
    _vector_dimension: int = PrivateAttr()
    _jag: JaguarHttpClient = PrivateAttr()
    _token: str = PrivateAttr()


    def __del__(self) -> None:
        pass

    @classmethod
    def class_name(cls) -> str:
        return "JaguarVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._jag

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings
        """
        use_node_metadata = add_kwargs.get("use_node_metadata", False)
        ids = []
        for node in nodes:
            text = node.get_text()
            embedding = node.get_embedding()
            if use_node_metadata is True:
                metadata = node.metadata
            else:
                metadata = None
            zid = self.add_text(text, embedding, metadata, **add_kwargs)
            ids.append(zid)

        return ids


    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query: VectorStoreQuery object
            kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        """
        embedding = query.query_embedding
        k = query.similarity_top_k
        (nodes, ids, simscores) = self.similarity_search_with_score(
            embedding, k=k, form="node", **kwargs
        )
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=simscores)

    def load_documents(
        self, embedding: List[float], k: int, **kwargs: Any
    ) -> List[Document]:
        """Query index to load top k most similar documents.

        Args:
            embedding: a list of floats
            k: topK number
            kwargs:  may contain 'where', 'metadata_fields', 'args', 'fetch_k'
        """
        return cast(
            List[Document],
            self.similarity_search_with_score(embedding, k=k, form="doc", **kwargs),
        )



    def is_anomalous(
        self,
        node: BaseNode,
        **kwargs: Any,
    ) -> bool:
        """Detect if given text is anomalous from the dataset.

        Args:
            query: Text to detect if it is anomaly
        Returns:
            True or False
        """
        vcol = self._vector_index
        vtype = self._vector_type
        str_embeddings = [str(f) for f in node.get_embedding()]
        qv_comma = ",".join(str_embeddings)
        podstore = self._pod + "." + self._store
        q = "select anomalous(" + vcol + ", '" + qv_comma + "', 'type=" + vtype + "')"
        q += " from " + podstore

        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return False
        jd = json.loads(js[0])
        if jd["anomalous"] == "YES":
            return True
        return False

    def run(self, query: str, withFile: bool = False) -> dict:
        """Run any query statement in jaguardb.

        Args:
            query (str): query statement to jaguardb
        Returns:
            None for invalid token, or
            json result string
        """
        if self._token == "":
            logger.error(f"E0005 error run({query})")
            return {}

        resp = self._jag.post(query, self._token, withFile)
        txt = resp.text
        try:
            return json.loads(txt)
        except Exception:
            return {}

    def count(self) -> int:
        """Count records of a store in jaguardb.

        Args: no args
        Returns: (int) number of records in pod store
        """
        podstore = self._pod + "." + self._store
        q = "select count() from " + podstore
        js = self.run(q)
        if isinstance(js, list) and len(js) == 0:
            return 0
        jd = json.loads(js[0])
        return int(jd["data"])

    def clear(self) -> None:
        """Delete all records in jaguardb.

        Args: No args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "truncate store " + podstore
        self.run(q)

    def drop(self) -> None:
        """Drop or remove a store in jaguardb.

        Args: no args
        Returns: None
        """
        podstore = self._pod + "." + self._store
        q = "drop store " + podstore
        self.run(q)

    def prt(self, msg: str) -> None:
        nows = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("/tmp/debugjaguar.log", "a") as file:
            print(f"{nows} msg={msg}", file=file, flush=True)

    def login(
        self,
        jaguar_api_key: Optional[str] = "",
    ) -> bool:
        """Login to jaguar server with a jaguar_api_key or let self._jag find a key.

        Args:
            optional jaguar_api_key (str): API key of user to jaguardb server
        Returns:
            True if successful; False if not successful
        """
        if jaguar_api_key == "":
            jaguar_api_key = self._jag.getApiKey()
        self._jaguar_api_key = jaguar_api_key
        self._token = self._jag.login(jaguar_api_key)
        if self._token == "":
            logger.error("E0001 error init(): invalid jaguar_api_key")
            return False
        return True

    def logout(self) -> None:
        """Logout to cleanup resources.

        Args: no args
        Returns: None
        """
        self._jag.logout(self._token)

    def _parseMeta(self, nvmap: dict, filecol: str) -> Tuple[List[str], List[str], str]:
        filepath = ""
        if filecol == "":
            nvec = list(nvmap.keys())
            vvec = list(nvmap.values())
        else:
            nvec = []
            vvec = []
            if filecol in nvmap:
                nvec.append(filecol)
                vvec.append(nvmap[filecol])
                filepath = nvmap[filecol]

            for k, v in nvmap.items():
                if k != filecol:
                    nvec.append(k)
                    vvec.append(v)

        return nvec, vvec, filepath
