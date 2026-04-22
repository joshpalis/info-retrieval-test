import os
import json
import csv
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class BRIGHTConverter:
    """
    Downloads BRIGHT parquet files from HuggingFace and converts to BEIR format.
    """

    AVAILABLE_TASKS = [
        "biology", "earth_science", "economics", "psychology",
        "robotics", "stackoverflow", "sustainable_living",
        "leetcode", "pony", "aops",
        "theoremqa_theorems", "theoremqa_questions"
    ]

    BASE_URL = "https://huggingface.co/datasets/xlangai/BRIGHT/resolve/main"

    def __init__(
        self,
        task: str,
        output_dir: str,
        high_relevance_score: int = 2,
        low_relevance_score: int = 1,
        prefix_ids: bool = False
    ):
        if task not in self.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task '{task}'. Available: {self.AVAILABLE_TASKS}"
            )
        self.task = task
        self.output_dir = output_dir
        self.high_rel = high_relevance_score
        self.low_rel = low_relevance_score
        self.prefix_ids = prefix_ids

    def convert(self) -> str:
        logger.info(f"Converting BRIGHT task '{self.task}' to BEIR format...")
        os.makedirs(os.path.join(self.output_dir, "qrels"), exist_ok=True)

        # Download parquet files
        documents_df = self._download_parquet("documents")
        examples_df = self._download_parquet("examples")

        # Log column names for debugging
        logger.info(f"Document columns: {list(documents_df.columns)}")
        logger.info(f"Example columns: {list(examples_df.columns)}")
        logger.info(f"Document sample:\n{documents_df.head(1).to_dict('records')}")
        logger.info(f"Example sample:\n{examples_df.head(1).to_dict('records')}")

        # Write corpus
        corpus_path = os.path.join(self.output_dir, "corpus.jsonl")
        doc_count = self._write_corpus(documents_df, corpus_path)
        logger.info(f"Wrote {doc_count} documents to {corpus_path}")

        # Write queries + qrels
        queries_path = os.path.join(self.output_dir, "queries.jsonl")
        qrels_path = os.path.join(self.output_dir, "qrels", "test.tsv")
        query_count, qrel_count = self._write_queries_and_qrels(
            examples_df, queries_path, qrels_path
        )
        logger.info(f"Wrote {query_count} queries to {queries_path}")
        logger.info(f"Wrote {qrel_count} qrel judgments to {qrels_path}")

        self._write_metadata()
        return self.output_dir

    def _download_parquet(self, folder: str) -> pd.DataFrame:
        """Download a parquet file from HuggingFace, with local caching."""
        cache_dir = os.path.join(self.output_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)

        filename = f"{self.task}-00000-of-00001.parquet"
        cache_path = os.path.join(cache_dir, f"{folder}_{filename}")

        if not os.path.exists(cache_path):
            url = f"{self.BASE_URL}/{folder}/{filename}"
            logger.info(f"Downloading: {url}")

            response = requests.get(url, stream=True, timeout=300)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download {url}: HTTP {response.status_code}"
                )

            # Stream download for large files
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"  Downloaded {downloaded // (1024*1024)}MB / "
                                    f"{total_size // (1024*1024)}MB ({pct:.0f}%)")

            logger.info(f"Saved to cache: {cache_path}")
        else:
            logger.info(f"Using cached: {cache_path}")

        # Read parquet — tries fastparquet first, then pyarrow
        try:
            df = pd.read_parquet(cache_path, engine='fastparquet')
        except Exception:
            try:
                df = pd.read_parquet(cache_path, engine='pyarrow')
            except Exception:
                df = pd.read_parquet(cache_path)

        logger.info(f"Loaded {len(df)} rows from {folder}/{self.task}")
        return df

    def _make_doc_id(self, raw_id: str) -> str:
        if self.prefix_ids:
            return f"{self.task}__{raw_id}"
        return raw_id

    def _make_query_id(self, raw_id: str) -> str:
        if self.prefix_ids:
            return f"{self.task}__{raw_id}"
        return raw_id

    def _write_corpus(self, df: pd.DataFrame, corpus_path: str) -> int:
        """Convert BRIGHT documents DataFrame to BEIR corpus.jsonl."""
        count = 0

        # Detect column names (BRIGHT may use 'id', '_id', 'content', 'text', etc.)
        id_col = self._find_column(df, ["id", "_id", "doc_id", "docid"])
        text_col = self._find_column(df, ["content", "text", "passage", "body"])
        title_col = self._find_column(df, ["title"], required=False)

        logger.info(f"Corpus mapping: id={id_col}, text={text_col}, title={title_col}")

        with open(corpus_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                doc_id = self._make_doc_id(str(row[id_col]))
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                title = ""
                if title_col and pd.notna(row.get(title_col)):
                    title = str(row[title_col])

                beir_doc = {
                    "_id": doc_id,
                    "title": title,
                    "text": text,
                    "metadata": {}
                }
                f.write(json.dumps(beir_doc) + '\n')
                count += 1

        return count

    def _write_queries_and_qrels(
        self, df: pd.DataFrame, queries_path: str, qrels_path: str
    ) -> tuple:
        """Convert BRIGHT examples DataFrame to BEIR queries.jsonl and qrels/test.tsv."""
        query_count = 0
        qrel_count = 0

        # Detect column names
        id_col = self._find_column(df, ["id", "_id", "query_id", "qid"])
        query_col = self._find_column(df, ["query", "text", "question"])
        gold_col = self._find_column(df, ["gold_ids", "gold_doc_ids"], required=False)
        gold_long_col = self._find_column(df, ["gold_ids_long", "gold_doc_ids_long"], required=False)

        logger.info(f"Query mapping: id={id_col}, query={query_col}, "
                     f"gold_ids={gold_col}, gold_ids_long={gold_long_col}")

        with open(queries_path, 'w', encoding='utf-8') as qf, \
             open(qrels_path, 'w', encoding='utf-8', newline='') as rf:

            writer = csv.writer(rf, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["query-id", "corpus-id", "score"])

            for _, row in df.iterrows():
                query_id = self._make_query_id(str(row[id_col]))
                query_text = str(row[query_col])

                beir_query = {"_id": query_id, "text": query_text, "metadata": {}}
                qf.write(json.dumps(beir_query) + '\n')
                query_count += 1

                seen_doc_ids = set()

                # High relevance
                if gold_col:
                    gold_ids = self._parse_ids(row.get(gold_col))
                    for doc_id in gold_ids:
                        doc_id = self._make_doc_id(str(doc_id))
                        if doc_id not in seen_doc_ids:
                            writer.writerow([query_id, doc_id, self.high_rel])
                            seen_doc_ids.add(doc_id)
                            qrel_count += 1

                # Lower relevance
                if gold_long_col:
                    gold_ids_long = self._parse_ids(row.get(gold_long_col))
                    for doc_id in gold_ids_long:
                        doc_id = self._make_doc_id(str(doc_id))
                        if doc_id not in seen_doc_ids:
                            writer.writerow([query_id, doc_id, self.low_rel])
                            seen_doc_ids.add(doc_id)
                            qrel_count += 1

        return query_count, qrel_count

    def _find_column(self, df: pd.DataFrame, candidates: list, required: bool = True) -> str:
        """Find the first matching column name from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col

        if required:
            raise ValueError(
                f"Could not find any of {candidates} in columns: {list(df.columns)}"
            )
        return None

    def _parse_ids(self, ids) -> list:
        """Handle gold_ids that may be list, JSON string, numpy array, or None."""
        if ids is None:
            return []
        if isinstance(ids, str):
            try:
                ids = json.loads(ids)
            except json.JSONDecodeError:
                return []
        if isinstance(ids, (list, tuple)):
            return list(ids)
        # Handle numpy arrays
        try:
            return list(ids)
        except TypeError:
            return []

    def _write_metadata(self):
        meta = {
            "source": "BRIGHT (xlangai/BRIGHT)",
            "task": self.task,
            "high_relevance_score": self.high_rel,
            "low_relevance_score": self.low_rel,
            "format": "BEIR-compatible",
            "relevance_mapping": {
                "gold_ids": f"score={self.high_rel}",
                "gold_ids_long": f"score={self.low_rel}",
                "all_other_docs": "score=0 (implicit)"
            }
        }
        meta_path = os.path.join(self.output_dir, "bright_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


def convert_bright_task(task: str, base_output_dir: str, **kwargs) -> str:
    """
    Convenience function to convert a single BRIGHT task.

    Usage:
        data_path = convert_bright_task("stackoverflow", "./beir_datasets")
    """
    output_dir = os.path.join(base_output_dir, f"bright-{task}")
    converter = BRIGHTConverter(task=task, output_dir=output_dir, **kwargs)
    return converter.convert()