"""Obsidian reader class.

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
from llama_index.readers.file import MarkdownReader
from llama_index.core.schema import Document


class ObsidianReader(BaseReader):
    """
    input_dir (str): Path to the Obsidian vault.
    extract_tasks (bool): If True, extract tasks from the text and store them in metadata.
                            Default is False.
    remove_tasks_from_text (bool): If True and extract_tasks is True, remove the task
                                    lines from the main document text.
                                    Default is False.
    """

    def __init__(
        self,
        input_dir: str,
        extract_tasks: bool = False,
        remove_tasks_from_text: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.extract_tasks = extract_tasks
        self.remove_tasks_from_text = remove_tasks_from_text


    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
        """
        Loads data in the LangChain document format.
        """
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]

    def _extract_wikilinks(self, text: str) -> List[str]:
        """
        Extracts Obsidian wikilinks from the given text.

        Matches patterns like:
          - [[Note Name]]
          - [[Note Name|Alias]]

        Returns a list of unique wikilink targets (aliases are ignored).
        """
        pattern = r"\[\[([^\]]+)\]\]"
        matches = re.findall(pattern, text)
        links = []
        for match in matches:
            # If a pipe is present (e.g. [[Note|Alias]]), take only the part before it.
            target = match.split("|")[0].strip()
            links.append(target)
        return list(set(links))

    def _extract_tasks(self, text: str) -> Tuple[List[str], str]:
        """
        Extracts markdown tasks from the text.

        A task is expected to be a checklist item in markdown, for example:
            - [ ] Do something
            - [x] Completed task

        Returns a tuple:
            (list of task strings, text with task lines removed if removal is enabled).
        """
        # This regex matches lines starting with '-' or '*' followed by a checkbox.
        task_pattern = re.compile(
            r"^\s*[-*]\s*\[\s*(?:x|X| )\s*\]\s*(.*)$", re.MULTILINE
        )
        tasks = task_pattern.findall(text)
        cleaned_text = text
        if self.remove_tasks_from_text:
            cleaned_text = task_pattern.sub("", text)
        return tasks, cleaned_text
