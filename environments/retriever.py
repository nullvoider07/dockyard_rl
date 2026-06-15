import sys
from typing import Any, NotRequired, Optional, TypedDict
import ray
import torch
from dockyard_rl.data.interfaces import LLMMessageLogType
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)

class RAGEnvConfig(TypedDict):
    num_workers: int
    # HuggingFace dataset to index for retrieval.
    # Defaults to "wikimedia/wikipedia" with subset "20231101.en".
    dataset_name: NotRequired[str]
    dataset_subset: NotRequired[Optional[str]]
    dataset_split: NotRequired[str]
    # Column in the dataset to index (must be a text field).
    text_column: NotRequired[str]
    # Maximum number of documents to return per retrieval query.
    top_k: NotRequired[int]
    # Maximum character length of each retrieved passage returned to the model.
    max_doc_length: NotRequired[int]
    stop_strings: NotRequired[Optional[list[str]]]

@ray.remote  # pragma: no cover
class BM25Retriever:
    """BM25-based sparse retriever backed by a HuggingFace dataset corpus.

    Builds a BM25 index over the ``text_column`` of the specified HuggingFace
    dataset at construction time. Retrieval is synchronous and runs on the same
    CPU worker process.

    Dependencies: ``rank_bm25``, ``transformers`` (for BERT tokenisation),
    ``datasets`` (for corpus loading).
    """

    def __init__(
        self,
        dataset_name: str = "wikimedia/wikipedia",
        dataset_subset: Optional[str] = "20231101.en",
        dataset_split: str = "train",
        text_column: str = "text",
        top_k: int = 3,
        max_doc_length: int = 500,
    ) -> None:
        from datasets import load_dataset
        from rank_bm25 import BM25Okapi
        from transformers import AutoTokenizer

        self.text_column = text_column
        self.top_k = top_k
        self.max_doc_length = max_doc_length

        print(
            f"[BM25Retriever] Loading corpus: {dataset_name} "
            f"(subset={dataset_subset}, split={dataset_split}) …"
        )
        ds = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        self.corpus: list[str] = ds[text_column]

        print(f"[BM25Retriever] Tokenising {len(self.corpus):,} documents …")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenised = [
            tokenizer.tokenize(doc[: self.max_doc_length]) for doc in self.corpus
        ]
        self.bm25 = BM25Okapi(tokenised)
        self._tokenizer = tokenizer
        print(f"[BM25Retriever] Index ready ({len(self.corpus):,} documents).")

    def retrieve(self, queries: list[str]) -> list[str]:
        """Retrieve and format top-k passages for each query.

        Args:
            queries: List of query strings.

        Returns:
            List of formatted retrieval strings, one per query. Each string
            contains the top-k passages separated by double newlines.
        """
        results = []
        for query in queries:
            tokens = self._tokenizer.tokenize(query)
            scores = self.bm25.get_scores(tokens)
            top_indices = scores.argsort()[-self.top_k :][::-1]
            passages = [
                self.corpus[i][: self.max_doc_length] for i in top_indices
            ]
            formatted = "\n\n".join(
                f"[Document {j + 1}]\n{p}" for j, p in enumerate(passages)
            )
            results.append(formatted)
        return results

class RAGEnvironmentMetadata(TypedDict):
    ground_truth: str

@ray.remote(  # type: ignore[call-overload]
    max_restarts=-1, max_task_retries=-1, max_concurrency=1000
)  # pragma: no cover
class RAGEnvironment(EnvironmentInterface[RAGEnvironmentMetadata]):
    """Multi-turn RAG (retrieval-augmented generation) environment.

    The agent triggers retrieval by emitting text inside ``<retrieve>…</retrieve>``
    tags. The environment extracts the query, fetches BM25 results, and returns
    them as an ``"environment"`` role observation. The episode continues until the
    agent emits a turn without a retrieval tag (i.e. produces a final answer).

    Rewards are always ``0.0`` — retrieval is a tool, not a graded action. Scoring
    the final answer is the responsibility of a downstream reward signal (e.g.
    ``MathEnvironment`` or ``RewardModelEnvironment``).

    Stop strings: ``["</retrieve>"]`` — generation halts as soon as the closing
    tag is emitted so the environment can process the query.
    """

    STOP_STRINGS: list[str] = ["</retrieve>"]

    def __init__(self, cfg: RAGEnvConfig) -> None:
        self.cfg = cfg
        self.num_workers = cfg.get("num_workers", 1)

        dataset_name = cfg.get("dataset_name", "wikimedia/wikipedia")
        dataset_subset = cfg.get("dataset_subset", "20231101.en")
        dataset_split = cfg.get("dataset_split", "train")
        text_column = cfg.get("text_column", "text")
        top_k = cfg.get("top_k", 3)
        max_doc_length = cfg.get("max_doc_length", 500)

        self.workers = [
            BM25Retriever.options(  # type: ignore[attr-defined]
                runtime_env={"py_executable": sys.executable}
            ).remote(
                dataset_name=dataset_name,
                dataset_subset=dataset_subset,
                dataset_split=dataset_split,
                text_column=text_column,
                top_k=top_k,
                max_doc_length=max_doc_length,
            )
            for _ in range(self.num_workers)
        ]
        self._worker_idx = 0

    def shutdown(self) -> None:
        for worker in self.workers:
            ray.kill(worker)  # type: ignore[arg-type]

    def _extract_retrieve_query(self, text: str) -> Optional[str]:
        """Extract the query inside the last <retrieve>…</retrieve> tags.

        Returns ``None`` if no tag is present (episode should terminate).
        """
        import re

        match = re.search(r"<retrieve>([\s\S]*?)(?:</retrieve>|$)", text)
        return match.group(1).strip() if match else None

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[RAGEnvironmentMetadata],
    ) -> EnvironmentReturn[RAGEnvironmentMetadata]:
        """Process a batch of model turns and return retrieval results.

        For each conversation:
        - If the last assistant turn contains ``<retrieve>…</retrieve>``, extract
          the query, retrieve documents, and return them as a new environment turn.
          The episode continues (``terminated = False``).
        - Otherwise, the agent has produced a final answer; the episode terminates
          (``terminated = True``) with zero reward.

        Args:
            message_log_batch: Batch of conversation message logs.
            metadata: Per-sample metadata (ground truth for downstream scoring).

        Returns:
            EnvironmentReturn with retrieval observations, zero rewards, and
            ``next_stop_strings`` set to ``STOP_STRINGS`` for ongoing episodes.
        """
        queries: list[Optional[str]] = []
        for conversation in message_log_batch:
            last_assistant = next(
                (
                    str(m["content"])
                    for m in reversed(conversation)
                    if m["role"] == "assistant"
                ),
                "",
            )
            queries.append(self._extract_retrieve_query(last_assistant))

        # Batch the non-None queries to a retriever worker (round-robin).
        active_indices = [i for i, q in enumerate(queries) if q is not None]
        active_queries = [queries[i] for i in active_indices]

        retrieved_docs: dict[int, str] = {}
        if active_queries:
            worker = self.workers[self._worker_idx % self.num_workers]
            self._worker_idx += 1
            docs: list[str] = ray.get(worker.retrieve.remote(active_queries))  # type: ignore[union-attr]
            for idx, doc in zip(active_indices, docs):
                retrieved_docs[idx] = doc

        observations = []
        terminateds_list = []
        next_stop_strings = []

        for i in range(len(message_log_batch)):
            if i in retrieved_docs:
                observations.append(
                    {
                        "role": "environment",
                        "content": f"Retrieved documents:\n\n{retrieved_docs[i]}",
                    }
                )
                terminateds_list.append(False)
                next_stop_strings.append(self.STOP_STRINGS)
            else:
                observations.append(
                    {
                        "role": "environment",
                        "content": "Environment: no retrieval query found. Generating final answer.",
                    }
                )
                terminateds_list.append(True)
                next_stop_strings.append(None)

        rewards = torch.zeros(len(message_log_batch), dtype=torch.float32).cpu()
        terminateds = torch.tensor(terminateds_list, dtype=torch.bool).cpu()

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute aggregate retrieval metrics for the completed batch.

        Retrieval rewards are always zero; the interesting metrics are turn counts
        and retrieval utilisation rate.

        Args:
            batch: Rollout batch. Expected keys: ``"is_end"``, ``"generation_lengths"``,
                ``"prompt_lengths"``.

        Returns:
            Tuple of (unmodified batch, metrics dict).
        """
        metrics: dict[str, float | int] = {
            "rag/fraction_properly_ended": batch["is_end"].float().mean().item(),
            "rag/num_samples": int(batch["is_end"].shape[0]),
            "rag/generation_lengths": batch["generation_lengths"].float().mean().item(),
            "rag/prompt_lengths": batch["prompt_lengths"].float().mean().item(),
        }
        return batch, metrics