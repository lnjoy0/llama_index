import asyncio
import os
import shlex
import shutil
from argparse import ArgumentParser
from glob import iglob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    Response,
    StreamingResponse,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, field_validator
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_pipeline.components.function import FnComponent
from llama_index.core.query_pipeline.query import QueryPipeline
from llama_index.core.readers.base import BaseReader
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.utils import get_cache_dir


def _try_load_openai_llm():
    try:
        from llama_index.llms.openai import OpenAI  # pants: no-infer-dep

        return OpenAI(model="gpt-3.5-turbo", streaming=True)
    except ImportError:
        raise ImportError(
            "`llama-index-llms-openai` package not found, "
            "please run `pip install llama-index-llms-openai`"
        )


RAG_HISTORY_FILE_NAME = "files_history.txt"


def default_ragcli_persist_dir() -> str:
    return str(Path(get_cache_dir()) / "rag_cli")


def query_input(query_str: Optional[str] = None) -> str:
    return query_str or ""


class QueryPipelineQueryEngine(CustomQueryEngine):
    query_pipeline: QueryPipeline = Field(
        description="Query Pipeline to use for Q&A.",
    )

    def custom_query(self, query_str: str) -> RESPONSE_TYPE:
        return self.query_pipeline.run(query_str=query_str)

    async def acustom_query(self, query_str: str) -> RESPONSE_TYPE:
        return await self.query_pipeline.arun(query_str=query_str)


class RagCLI(BaseModel):
    """
    CLI tool for chatting with output of a IngestionPipeline via a QueryPipeline.
    """

    ingestion_pipeline: IngestionPipeline = Field(
        description="Ingestion pipeline to run for RAG ingestion."
    )
    verbose: bool = Field(
        description="Whether to print out verbose information during execution.",
        default=False,
    )
    persist_dir: str = Field(
        description="Directory to persist ingestion pipeline.",
        default_factory=default_ragcli_persist_dir,
    )
    llm: LLM = Field(
        description="Language model to use for response generation.",
        default_factory=lambda: _try_load_openai_llm(),
    )
    query_pipeline: Optional[QueryPipeline] = Field(
        description="Query Pipeline to use for Q&A.",
        default=None,
    )
    chat_engine: Optional[CondenseQuestionChatEngine] = Field(
        description="Chat engine to use for chatting.",
        default=None,
    )
    file_extractor: Optional[Dict[str, BaseReader]] = Field(
        description="File extractor to use for extracting text from files.",
        default=None,
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator("query_pipeline", mode="before")
    def query_pipeline_from_ingestion_pipeline(
        cls, query_pipeline: Any, values: Dict[str, Any]
    ) -> Optional[QueryPipeline]:
        """
        If query_pipeline is not provided, create one from ingestion_pipeline.
        """
        if query_pipeline is not None:
            return query_pipeline

        ingestion_pipeline = cast(IngestionPipeline, values["ingestion_pipeline"])
        if ingestion_pipeline.vector_store is None:
            return None
        verbose = cast(bool, values["verbose"])
        query_component = FnComponent(
            fn=query_input, output_key="output", req_params={"query_str"}
        )
        llm = cast(LLM, values["llm"])

        # get embed_model from transformations if possible
        embed_model = None
        if ingestion_pipeline.transformations is not None:
            for transformation in ingestion_pipeline.transformations:
                if isinstance(transformation, BaseEmbedding):
                    embed_model = transformation
                    break

        Settings.llm = llm
        Settings.embed_model = embed_model

        retriever = VectorStoreIndex.from_vector_store(
            ingestion_pipeline.vector_store,
        ).as_retriever(similarity_top_k=8)
        response_synthesizer = CompactAndRefine(streaming=True, verbose=verbose)

        # define query pipeline
        query_pipeline = QueryPipeline(verbose=verbose)
        query_pipeline.add_modules(
            {
                "query": query_component,
                "retriever": retriever,
                "summarizer": response_synthesizer,
            }
        )
        query_pipeline.add_link("query", "retriever")
        query_pipeline.add_link("retriever", "summarizer", dest_key="nodes")
        query_pipeline.add_link("query", "summarizer", dest_key="query_str")
        return query_pipeline


    async def handle_question(self, question: str) -> None:
        if self.query_pipeline is None:
            raise ValueError("query_pipeline is not defined.")
        query_pipeline = cast(QueryPipeline, self.query_pipeline)
        query_pipeline.verbose = self.verbose
        chat_engine = cast(CondenseQuestionChatEngine, self.chat_engine)
        response = chat_engine.chat(question)

        if isinstance(response, StreamingResponse):
            response.print_response_stream()
        else:
            response = cast(Response, response)
            print(response)

    async def start_chat_repl(self) -> None:
        """
        Start a REPL for chatting with the agent.
        """
        if self.query_pipeline is None:
            raise ValueError("query_pipeline is not defined.")
        chat_engine = cast(CondenseQuestionChatEngine, self.chat_engine)
        chat_engine.streaming_chat_repl()

    @classmethod
    def add_parser_args(
        cls,
        parser: Union[ArgumentParser, Any],
        instance_generator: Optional[Callable[[], "RagCLI"]],
    ) -> None:
        if instance_generator:
            parser.add_argument(
                "-q",
                "--question",
                type=str,
                help="The question you want to ask.",
                required=False,
            )

            parser.add_argument(
                "-f",
                "--files",
                type=str,
                nargs="+",
                help=(
                    "The name of the file(s) or directory you want to ask a question about,"
                    'such as "file.pdf". Supports globs like "*.py".'
                ),
            )
            parser.add_argument(
                "-c",
                "--chat",
                help="If flag is present, opens a chat REPL.",
                action="store_true",
            )
            parser.add_argument(
                "-v",
                "--verbose",
                help="Whether to print out verbose information during execution.",
                action="store_true",
            )
            parser.add_argument(
                "--clear",
                help="Clears out all currently embedded data.",
                action="store_true",
            )
            parser.add_argument(
                "--create-llama",
                help="Create a LlamaIndex application with your embedded data.",
                required=False,
                action="store_true",
            )
            parser.set_defaults(
                func=lambda args: asyncio.run(
                    instance_generator().handle_cli(**vars(args))
                )
            )

    def cli(self) -> None:
        """
        Entrypoint for CLI tool.
        """
        parser = ArgumentParser(description="LlamaIndex RAG Q&A tool.")
        subparsers = parser.add_subparsers(
            title="commands", dest="command", required=True
        )
        llamarag_parser = subparsers.add_parser(
            "rag", help="Ask a question to a document / a directory of documents."
        )
        self.add_parser_args(llamarag_parser, lambda: self)
        # Parse the command-line arguments
        args = parser.parse_args()

        # Call the appropriate function based on the command
        args.func(args)
