"""Pairwise entity reranker: are these two nodes one real-world entity?

The merge queue the workbench shows is ranked by hand rules - name equivalence,
a structure-aware Levenshtein, an embedding shortlist - and those rules get
patched every week (six matching fixes on 2026-09-01 alone), because each is a
comparison of NAMES and a name is thin evidence: "Tic Tac Sighting" and "2004
USS Nimitz UAP encounter" share no word, while "Apollo 11" and "Apollo 12"
share all but one character. What a reviewer actually reads before deciding is
the two names, the two types, and a few claims from each side. This scores that
reading with a cross-encoder so the queue is ordered by it.

The model is Qwen3-Reranker-0.6B, a causal LM prompted as a yes/no judge; the
score is its probability of "yes" at the last position. It is a RANKER, not a
decider: nothing here merges, and cross-type pairs still go to the Claude
verify pass, because a differing type is weak evidence against identity and a
cross-type name match is more often two referents than one (node-types.md).

Runs on the GPU when the container has one, else CPU. The weights live outside
the image (ENTITY_RERANKER_MODEL_PATH, default under the assimilator data
directory) because 1.2 GB does not belong in a build layer.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# The model id is the policy's vocabulary (model-policy.yaml, stage `rerank`);
# the weights live outside the image under the assimilator data directory.
# ENTITY_RERANKER_MODEL_PATH overrides the directory outright (the container
# mounts the data directory somewhere else).
DEFAULT_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
_WEIGHTS_DIR = {DEFAULT_MODEL_ID: "qwen3-reranker-0.6b"}
MODELS_ROOT = Path(
    os.environ.get(
        "ASSIMILATOR_MODELS_DIR",
        str(Path.home() / ".local" / "share" / "assimilator" / "models"),
    )
)


def weights_path(model_id: str = DEFAULT_MODEL_ID) -> Path:
    """Where a policy model id's weights are expected on disk."""
    override = os.environ.get("ENTITY_RERANKER_MODEL_PATH")
    if override:
        return Path(override)
    sub = _WEIGHTS_DIR.get(model_id, model_id.replace("/", "--").lower())
    return MODELS_ROOT / sub


DEFAULT_MODEL_PATH = weights_path()

# The question the model answers. Named relations that a name comparison
# confuses with identity are spelled out as NOT the same entity, because those
# are the pairs the rules get wrong - the Reid conflation was a surname match,
# the Nimitz split had a ship, an incident and a video on three nodes.
INSTRUCTION = (
    "Given two entries from a knowledge graph about anomalous phenomena, judge "
    "whether they refer to the same real-world entity: the same person, "
    "organisation, project, event, place, object, document or topic, possibly "
    "under different names or spellings. Related but distinct things are NOT the "
    "same entity: a person and their organisation, a parent and a child, a ship "
    "and an incident aboard it, a mission and one event during it, a report and "
    "the investigation that produced it, two numbered missions in one programme."
)

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be "yes" '
    'or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

CLAIMS_PER_ENTITY = 3
CLAIM_CHARS = 240


@dataclass
class Entity:
    """What the model sees of a node: as much as a reviewer reads."""

    name: str
    node_type: str
    claims: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"Name: {self.name}", f"Type: {self.node_type}"]
        if self.claims:
            lines.append("Claims:")
            lines += [f"- {c[:CLAIM_CHARS]}" for c in self.claims[:CLAIMS_PER_ENTITY]]
        else:
            lines.append("Claims: (none recorded)")
        return "\n".join(lines)


def entity_from_graph(conn: sqlite3.Connection, node_id: str) -> Entity | None:
    """A node with its first few claims, in a stable order (by rowid, so two
    runs over one graph score identically)."""
    row = conn.execute(
        "SELECT name, node_type FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    if row is None:
        return None
    claims = [
        r[0]
        for r in conn.execute(
            "SELECT c.content FROM claim_node_refs r JOIN claims c ON c.id = r.claim_id "
            "WHERE r.node_id = ? ORDER BY c.rowid LIMIT ?",
            (node_id, CLAIMS_PER_ENTITY),
        )
    ]
    return Entity(name=row[0], node_type=row[1], claims=claims)


def pair_prompt(a: Entity, b: Entity) -> str:
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {a.text()}\n<Document>: {b.text()}"


class EntityReranker:
    """Loads once, scores many. Construction is the expensive step (weights to
    the device); keep one instance per process."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        device: str | None = None,
        max_length: int = 1024,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        path = str(model_path or weights_path(model_id))
        if not Path(path).is_dir():
            raise FileNotFoundError(
                f"reranker weights for {model_id} not found at {path}; set "
                "ENTITY_RERANKER_MODEL_PATH or download the model there"
            )
        # ENTITY_RERANKER_DEVICE=cpu forces the CPU: the GPU is shared with the
        # embedding service and the digester's entailment model, and a 1.2 GB
        # load into 70 MB of free memory fails after the weights are read.
        self.device = (
            device
            or os.environ.get("ENTITY_RERANKER_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        # Memory-efficient attention (sdpa): eager attention materialises a
        # length-squared matrix per head and per layer, which bounds the batch on
        # a 6 GB card; and the run is memory-bound under the laptop's 20 W power
        # cap, so the batch is what amortises each read of the 1.2 GB of weights.
        # (The Triton kernel torch 2.13 uses for the rotary embedding needs gcc
        # and libc6-dev in the image, whatever the attention implementation.)
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, attn_implementation="sdpa"
            )
            .to(self.device)
            .eval()
        )
        self.max_length = max_length
        self._prefix = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        self._suffix = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)
        self._yes = self.tokenizer.convert_tokens_to_ids("yes")
        self._no = self.tokenizer.convert_tokens_to_ids("no")

    def _batch_scores(self, prompts: list[str]) -> list[float]:
        import torch

        room = self.max_length - len(self._prefix) - len(self._suffix)
        enc = self.tokenizer(
            prompts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=room,
        )
        enc["input_ids"] = [
            self._prefix + ids + self._suffix for ids in enc["input_ids"]
        ]
        batch = self.tokenizer.pad(
            enc, padding=True, return_tensors="pt", max_length=self.max_length
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            # Only the last position's logits are read, so only those are
            # computed: the full tensor is batch x length x a 152k-token
            # vocabulary, 1.8 GB at batch 32, and it was the out-of-memory.
            try:
                out = self.model(**batch, logits_to_keep=1)
            except TypeError:  # an older transformers without the argument
                out = self.model(**batch)
            logits = out.logits[:, -1, :]
            pair = torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1)
            return torch.log_softmax(pair.float(), dim=1)[:, 1].exp().tolist()

    def score(
        self,
        pairs: list[tuple[Entity, Entity]],
        batch_size: int = 8,
        symmetric: bool = True,
    ) -> list[float]:
        """P(same entity) per pair. symmetric scores both orders and averages:
        the model is a query/document judge and the two roles are not
        interchangeable, so a single order carries an ordering artefact."""
        prompts = [pair_prompt(a, b) for a, b in pairs]
        if symmetric:
            prompts += [pair_prompt(b, a) for a, b in pairs]
        # Batch by length so a batch pads to its own longest prompt, not the
        # corpus's: the work is proportional to the padded length.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        scores = [0.0] * len(prompts)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            for j, s in zip(idx, self._batch_scores([prompts[k] for k in idx])):
                scores[j] = s
        if not symmetric:
            return scores
        n = len(pairs)
        return [(scores[i] + scores[n + i]) / 2 for i in range(n)]

    def peak_memory_mb(self) -> int | None:
        """Peak device memory this process has held, for the run record."""
        import torch

        if self.device != "cuda":
            return None
        return round(torch.cuda.max_memory_allocated() / 2**20)


_instance: EntityReranker | None = None


def get_reranker(model_id: str = DEFAULT_MODEL_ID) -> EntityReranker:
    global _instance
    if _instance is None or _instance.model_id != model_id:
        _instance = EntityReranker(model_id=model_id)
    return _instance
