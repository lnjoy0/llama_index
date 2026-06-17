"""
Obsidian reader class.

Pass in the path to an Obsidian vault and it will parse all markdown
files into a List of Documents. Documents are split by header in
the Markdown Reader we use.

Each document will contain the following metadata:
- file_name: the name of the markdown file
- folder_path: the full path to the folder containing the file
- folder_name: the relative path to the folder containing the file
- note_name: the name of the note (without the .md extension)
- wikilinks: a list of all wikilinks found in the document
- backlinks: a list of all notes that link to this note

Optionally, tasks can be extracted from the text and stored in metadata.

"""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from langchain.docstore.document import Document as LCDocument

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.file import MarkdownReader


