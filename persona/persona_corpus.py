"""Prepare local, source-traceable persona-manager prompts without model calls.

Book text stays in the ignored local JSONL; the manifest contains metadata only.
EPUB offsets refer to normalized Unicode text within the named spine document,
not raw ZIP bytes. Mutations preserve source split/pair identity and explicitly
change an instruction or supported speaker attribution, never reverse tokens.
IPIP inventory items retain their original keys and stay outside adaptive fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import unquote

from persona_vectors import canonical_digest

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "persona":
    PROJECT_ROOT = PROJECT_ROOT.parent


SCHEMA = "character-creation.manager-corpus.v1"
CHARACTERS = ("Skippy", "Joe Bishop", "Joe", "Bishop")
BOOK_TITLES = ("columbusday", "specops", "paradise", "troubleonparadise", "blackops", "zerohour",
               "mavericks", "renegades", "armageddon", "valkyrie", "criticalmass")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(key: str, original: str | None = None, *, seed: int = 7035) -> str:
    if original is not None:
        aliases = {"discovery": "discovery", "train": "discovery", "training": "discovery",
                   "heldout": "heldout", "test": "heldout", "val": "heldout", "validation": "heldout", "inventory": "heldout"}
        if original not in aliases:
            raise ValueError(f"Unknown source split {original!r}")
        return aliases[original]
    return "heldout" if int(canonical_digest({"seed": seed, "group": key}), 16) % 5 == 0 else "discovery"


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden and tag in {"p", "div", "br", "li", "h1", "h2", "h3", "section", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {"p", "div", "li", "h1", "h2", "h3", "section", "blockquote"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def _zip_path(base: str, href: str) -> str:
    href = unquote(href.split("#", 1)[0], errors="strict")
    if ":" in href or href.startswith("/"):
        raise ValueError("EPUB member must be a relative local path")
    parts = []
    for part in PurePosixPath(base, href).parts:
        if part == "..":
            if not parts:
                raise ValueError("EPUB member escapes archive")
            parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    return "/".join(parts)


def book_documents(path: Path, *, max_document_bytes=16_000_000, max_total_bytes=128_000_000):
    """Yield (member, normalized text) in EPUB spine order, or exact UTF-8 TXT."""
    if path.suffix.lower() == ".txt":
        if path.stat().st_size > max_total_bytes:
            raise ValueError("Text book exceeds explicit extraction size bound")
        yield "", path.read_text(encoding="utf-8-sig")
        return
    if path.suffix.lower() != ".epub":
        raise ValueError("Book extraction supports explicit EPUB or UTF-8 TXT files")
    with zipfile.ZipFile(path) as archive:
        total = 0

        def read(member):
            nonlocal total
            info = archive.getinfo(member)
            total += info.file_size
            if info.file_size > max_document_bytes or total > max_total_bytes:
                raise ValueError("EPUB expanded document size exceeds explicit bound")
            return archive.read(member)

        container = ET.fromstring(read("META-INF/container.xml"))
        roots = [item.attrib["full-path"] for item in container.iter() if item.tag.rsplit("}", 1)[-1] == "rootfile"]
        if len(roots) != 1:
            raise ValueError("EPUB requires one unambiguous package root")
        package_path = _zip_path("", roots[0])
        package = ET.fromstring(read(package_path))
        manifest = {item.attrib["id"]: item.attrib for item in package.iter() if item.tag.rsplit("}", 1)[-1] == "item"}
        spine = [item.attrib["idref"] for item in package.iter() if item.tag.rsplit("}", 1)[-1] == "itemref"]
        if not spine or len(set(spine)) != len(spine):
            raise ValueError("EPUB spine is missing or repeats a document")
        for identity in spine:
            item = manifest[identity]
            if item.get("media-type") not in {"application/xhtml+xml", "text/html"}:
                raise ValueError("Unsupported non-HTML EPUB spine item")
            member = _zip_path(str(PurePosixPath(package_path).parent), item["href"])
            html = read(member).decode("utf-8-sig")
            parser = PlainHTML()
            parser.feed(html)
            yield member, parser.text()


def chunks(text: str, max_chars: int = 1600):
    """Nonoverlapping character ranges; every non-whitespace character is kept."""
    if type(max_chars) is not int or max_chars < 128:
        raise ValueError("Chunk size must be an integer of at least 128 characters")
    start = 0
    while start < len(text):
        stop = min(start + max_chars, len(text))
        if stop < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, stop)
            if boundary >= 0:
                stop = boundary + 1
        if text[start:stop].strip():
            yield start, stop, text[start:stop]
        start = stop


def dialogue_labels(text: str, characters=CHARACTERS) -> dict:
    """Conservative surface labels; mentions alone are never speaker attribution."""
    if not characters or any(not isinstance(name, str) or not name for name in characters):
        raise ValueError("Character allowlist must contain explicit nonempty names")
    names = "|".join(re.escape(name) for name in sorted(set(characters), key=lambda x: (-len(x), x)))
    mentions = {name: len(re.findall(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text)) for name in characters}
    mentions = {name: count for name, count in mentions.items() if count}
    spans = []
    colon = re.compile(r"(?m)^[ \t]*(?P<speaker>" + names + r"):[ \t]*(?P<dialogue>[^\n]+)")
    for match in colon.finditer(text):
        spans.append({"speaker": match["speaker"], "speaker_start": match.start("speaker"),
                      "speaker_end": match.end("speaker"), "dialogue_start": match.start("dialogue"),
                      "dialogue_end": match.end("dialogue"), "evidence": "explicit_colon_label"})
    tags = r"(?:said|asked|replied|answered|whispered|shouted|muttered)"
    quote = r"[\"“](?P<dialogue>[^\"”\n]{1,700})[\"”][ ,;.!?—-]*"
    for pattern in (quote + r"(?P<speaker>" + names + r")\s+" + tags + r"\b",
                    quote + tags + r"\s+(?P<speaker>" + names + r")(?!\w)"):
        for match in re.finditer(pattern, text):
            spans.append({"speaker": match["speaker"], "speaker_start": match.start("speaker"),
                          "speaker_end": match.end("speaker"), "dialogue_start": match.start("dialogue"),
                          "dialogue_end": match.end("dialogue"), "evidence": "explicit_adjacent_speech_tag"})
    unique = {(span["speaker_start"], span["speaker_end"]): span for span in spans}
    return {"character_mentions": mentions, "dialogue_spans": [unique[key] for key in sorted(unique)],
            "span_offset_basis": "Unicode codepoints within this source chunk; add source.char_start for document offsets",
            "attribution_method": "exact allowlisted names with colon labels or adjacent speech tags; unresolved narration stays unlabeled"}


def swap_speakers(text: str, spans: list[dict], *, counterfactual=False):
    names = sorted({span["speaker"] for span in spans})
    if not names or (len(names) < 2 and not counterfactual):
        return None
    # Surface aliases are not independent roles. Avoid treating Joe/Bishop as
    # different speakers; this guard does not relabel any observed attribution.
    joe_aliases = {"Joe", "Bishop", "Joe Bishop"}
    pairs = [(left, right) for i, left in enumerate(names) for right in names[i + 1:]
             if not ({left, right} <= joe_aliases)]
    if not pairs and not counterfactual:
        return None
    if counterfactual:
        left, right = "Skippy", "Joe Bishop"
        replacements = {"Skippy": "Joe Bishop", **{alias: "Skippy" for alias in joe_aliases}}
    else:
        left, right = pairs[0]
        replacements = {left: right, right: left}
    edits = []
    for span in spans:
        start, end, speaker = span["speaker_start"], span["speaker_end"], span["speaker"]
        if text[start:end] != speaker:
            raise ValueError("Speaker span does not match source characters")
        if speaker in replacements:
            edits.append({"start": start, "end": end, "original": speaker, "replacement": replacements[speaker]})
    if not edits:
        return None
    edits.sort(key=lambda row: row["start"])
    if any(a["end"] > b["start"] for a, b in zip(edits, edits[1:])):
        raise ValueError("Speaker substitutions overlap")
    changed = text
    for edit in reversed(edits):
        changed = changed[:edit["start"]] + edit["replacement"] + changed[edit["end"]:]
    return changed, {"kind": "speaker_attribution_counterfactual" if counterfactual else "speaker_attribution_swap",
                     "speakers": [left, right], "edits": edits,
                     "counterfactual": counterfactual,
                     "scope": "explicit speaker attributions only; dialogue words and narrative mentions unchanged"}


def source_info(path: Path, corpus: str) -> dict:
    return {"path": str(path.resolve()), "corpus": corpus, "sha256": sha256(path), "bytes": path.stat().st_size}


def record(source: dict, text: str, *, corpus: str, category: str, split: str, locator: dict,
           prompt: str | None = None, system="", mutation=None, labels=None, purpose="behavior", max_tokens=2048, **extra):
    source_id = canonical_digest({"sha256": source["sha256"], **locator})
    row = {"schema": SCHEMA, "corpus": corpus, "category": category, "split": split,
           "source_id": source_id, "source_sha256": source["sha256"], "source": {**source, **locator},
           "pair_id": source_id, "source_text_sha256": text_digest(text),
           "prompt": text if prompt is None else prompt, "system": system, "purpose": purpose,
           "max_tokens": max_tokens, "thinking": False, "mutation": mutation or {"kind": "original"},
           "labels": labels or {}, "fit_on_heldout": False, "token_budget_status": "requires_runtime_tokenizer_check", **extra}
    row["id"] = canonical_digest(row)
    return row


def book_records(path: Path, *, max_chars=1600, characters=CHARACTERS, seed=7035):
    source = source_info(path, "skippy")
    instruction = "Analyze the character traits and conversational dynamics supported by this passage. Identify uncertainty; do not invent dialogue or speaker attribution."
    negative = "Do not infer character traits from this passage. Instead, summarize its factual events and identify the limits of speaker attribution."
    for member, text in book_documents(path):
        document_sha = text_digest(text)
        # Whole source documents/chapters stay together; mutation arms cannot leak.
        split = split_for(document_sha, seed=seed)
        for start, end, excerpt in chunks(text, max_chars):
            locator = {"member": member, "document_text_sha256": document_sha, "char_start": start, "char_end": end,
                       "offset_basis": "Unicode codepoints in normalized spine document" if member else "Unicode codepoints in decoded UTF-8 TXT"}
            labels = dialogue_labels(excerpt, characters)
            common = dict(corpus="skippy", category="book_character_dialogue", split=split,
                          locator=locator, labels=labels, private_source_text=True)
            yield record(source, excerpt, prompt=instruction + "\n\nPassage:\n" + excerpt, **common)
            yield record(source, excerpt, prompt=negative + "\n\nPassage:\n" + excerpt,
                         mutation={"kind": "instruction_negation", "original_instruction": instruction,
                                   "replacement_instruction": negative, "scope": "trait inference instruction; passage unchanged"}, **common)
            swapped = swap_speakers(excerpt, labels["dialogue_spans"])
            if swapped is None:
                swapped = swap_speakers(excerpt, labels["dialogue_spans"], counterfactual=True)
            if swapped is not None:
                changed, mutation = swapped
                mutation_instruction = ("This is a deliberately counterfactual speaker-attribution edit, not the original passage. "
                    if mutation["counterfactual"] else "")
                yield record(source, excerpt, prompt=mutation_instruction + instruction + "\n\nPassage:\n" + changed,
                             mutation=mutation, **common)


def jsonl_records(path: Path, corpus: str, *, seed=7035, limit=None, mutations=True, max_prompt_chars=6000):
    source = source_info(path, corpus)
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if limit is not None and index > limit:
                break
            item = json.loads(line)
            text = item["prompt"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid source prompt at {path}:{index}")
            if len(text) > max_prompt_chars:
                continue  # Explicit bounded prompt-selection policy in the manifest.
            original_split = item.get("split")
            split = split_for(text_digest(text), original_split, seed=seed)
            locator = {"jsonl_line": index, "char_start": 0, "char_end": len(text),
                       "offset_basis": "Unicode codepoints in source prompt field", "original_split": original_split}
            labels = ({key: item[key] for key in ("issue_area", "expected_frames", "contrast_frames", "domain_frames") if key in item}
                      if corpus == "scotus" else {"source_category": item.get("category")})
            common = dict(corpus=corpus, category=item.get("issue_area", item.get("category", "unspecified")),
                          split=split, locator=locator, labels=labels,
                          max_tokens=3072 if corpus == "scotus" else 2048,
                          original_prompt_id=item.get("prompt_key", item.get("prompt_id", index)))
            yield record(source, text, **common)
            if mutations:
                original = "Answer the following task directly."
                opposite = "Do not answer the following task directly. Instead, explain how to approach it without giving its final answer."
                # The direct arm is explicit too, so the pair differs only in its instruction.
                yield record(source, text, prompt=original + "\n\nTask:\n" + text,
                             mutation={"kind": "instruction_direct", "instruction": original}, **common)
                yield record(source, text, prompt=opposite + "\n\nTask:\n" + text,
                             mutation={"kind": "instruction_negation", "original_instruction": original,
                                       "replacement_instruction": opposite, "scope": "task instruction only; source task unchanged"},
                             expected_answer=None, grade_expected_answer=False, **common)


def big_five_records(inventory_path: Path, tasks_path: Path):
    source = source_info(inventory_path, "big_five")
    inventory = json.loads(inventory_path.read_text())
    questions = inventory["questions"]
    if len(questions) != 50 or len({item["id"] for item in questions}) != 50:
        raise ValueError("IPIP inventory requires all fifty unique items")
    for item in questions:
        if type(item["reverse_scored"]) is not bool or item["dimension"] not in inventory["dimensions"]:
            raise ValueError("IPIP scoring keys must remain explicit")
        text = item["text"]
        prompt = "Rate how accurately this statement describes your usual style on a scale from 1 (very inaccurate) to 5 (very accurate). Reply with only one digit. Statement: " + text
        yield record(source, text, corpus="big_five", category="ipip50_inventory", split="heldout",
                     locator={"inventory_item_id": item["id"], "original_split": "inventory", "char_start": 0,
                              "char_end": len(text), "offset_basis": "Unicode codepoints in inventory item text"},
                     prompt=prompt, purpose="inventory_measurement", labels={"inventory_item": dict(item)},
                     adaptive_fit_allowed=False, mutation={"kind": "original_inventory_item"})
    task_source = source_info(tasks_path, "big_five")
    tasks = json.loads(tasks_path.read_text())
    profiles = [("baseline", "", None, None)]
    for trait, poles in sorted(tasks["trait_prompts"].items()):
        profiles.extend((f"{trait}_{pole}", prompt, trait, pole) for pole, prompt in sorted(poles.items()))
    for item in tasks["prompts"]:
        for profile, system, trait, pole in profiles:
            yield record(task_source, item["text"], corpus="big_five", category=item["category"],
                         split=split_for(item["id"], item["split"]),
                         locator={"prompt_id": item["id"], "original_split": item["split"], "char_start": 0,
                                  "char_end": len(item["text"]), "offset_basis": "Unicode codepoints in authored task prompt"},
                         system=system, purpose="prompt_conditioned_behavior", labels={"profile": profile,
                             "instructed_trait": trait, "instructed_pole": pole, "measured_trait": None},
                         reference_answer=item.get("reference_answer"), capability=item["capability"],
                         adaptive_fit_allowed=item["split"] == "discovery")


def _publish_manifest(path: Path, value: dict):
    """Atomic metadata publication; the temporary file never becomes an artifact."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def known_book_name(name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    return (any(marker in compact for marker in ("expeditionaryforce", "craigalanson", "skippy"))
            or any(re.fullmatch(r"\d*" + title, compact) for title in BOOK_TITLES))


def book_candidates(directory: Path) -> list[Path]:
    """Bounded drop-folder scan; Downloads includes only the owner's named books."""
    if not directory.is_dir():
        return []
    candidates = []
    downloads = directory.name.lower() == "downloads"
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            continue
        if path.is_file() and path.suffix.lower() in {".epub", ".txt"}:
            if not downloads or known_book_name(path.stem):
                candidates.append(path)
        elif path.is_dir() and known_book_name(path.name):
            # Only one child level, and only a specifically recognized book
            # folder. No torrents, PDFs, archives or unrelated download trees.
            candidates.extend(child for child in sorted(path.iterdir())
                              if not child.is_symlink() and child.is_file() and child.suffix.lower() in {".epub", ".txt"})
    return candidates


def prepare(root: Path, output: Path, manifest_path: Path, *, books=(), seed=7035,
            max_chars=1600, contrastive_limit=None, max_prompt_chars=6000, overwrite=False,
            include_contrastive=False, book_dir=None):
    """Write a local corpus and a text-free metadata manifest; no uploads or inference."""
    if output.resolve() == manifest_path.resolve() or (not overwrite and (output.exists() or manifest_path.exists())):
        raise FileExistsError("Corpus outputs already exist or collide; use explicit --overwrite for owned regeneration")
    if (type(seed) is not int or not 0 <= seed < 2**32 or type(max_chars) is not int or max_chars < 128
            or type(max_prompt_chars) is not int or max_prompt_chars < 128
            or (contrastive_limit is not None and (type(contrastive_limit) is not int or contrastive_limit < 1))):
        raise ValueError("Corpus seed and extraction limits must be explicit valid positive values")
    book_by_hash = {}
    for path in sorted(Path(path) for path in books):
        book_by_hash.setdefault(sha256(path), path)
    books = list(book_by_hash.values())
    known = [(root / "data/scotus/scotus_poke_prompts_v1.jsonl", "scotus"),
             (root / "data/big_five_ocean_test.json", "big_five"),
             (root / "data/persona_prompts.json", "big_five"),
             (root / "contrastive_data/prompts_100k.jsonl", "contrastive")]
    sources = [source_info(path, corpus) for path, corpus in known if path.is_file() and (corpus != "contrastive" or include_contrastive)]
    sources.extend(source_info(path, "skippy") for path in books)
    counts, mutation_counts, skipped_duplicates = Counter(), Counter(), 0
    ids, pair_splits = set(), {}
    manifest = {"schema": SCHEMA, "status": "preparing", "sources": sources,
                "output": str(output.resolve()), "split_seed": seed,
                "split_policy": "preserve supplied splits; otherwise fixed 80/20 content-hash grouping; EPUB document and all mutation arms stay together",
                "selection": {"max_book_chunk_chars": max_chars, "max_source_prompt_chars": max_prompt_chars,
                              "contrastive_line_limit": contrastive_limit},
                "books_status": "available" if books else "pending_redownload", "generation": 1,
                "book_dir": str((Path(book_dir) if book_dir else root / "books").resolve()),
                "book_scan_depth": 2, "book_scan_policy": "explicit directory plus one child level only for recognized Skippy book folders; Downloads direct files also require recognized title/author; EPUB/TXT only, no torrents or source copying",
                "include_contrastive": include_contrastive,
                "private_text_policy": "JSONL may contain copyrighted book passages; keep local and ignored by Git; do not publish",
                "inventory_policy": "IPIP50 text, IDs, dimensions and reverse keys preserved; separate heldout inventory measurement, never adaptive fitting",
                "claims": "source/character labels are observations or explicit source annotations, not validated model effects"}
    streams = [jsonl_records(known[0][0], "scotus", seed=seed, max_prompt_chars=max_prompt_chars),
               big_five_records(known[1][0], known[2][0])]
    if include_contrastive and known[3][0].is_file():
        streams.append(jsonl_records(known[3][0], "contrastive", seed=seed, limit=contrastive_limit, max_prompt_chars=max_prompt_chars))
    streams.extend(book_records(path, seed=seed, max_chars=max_chars) for path in books)
    try:
        with output.open("w" if overwrite else "x", encoding="utf-8") as destination:
            for stream in streams:
                for row in stream:
                    if row["id"] in ids:
                        skipped_duplicates += 1
                        continue
                    ids.add(row["id"])
                    if row["pair_id"] in pair_splits and pair_splits[row["pair_id"]] != row["split"]:
                        raise ValueError("A source/mutation pair crosses discovery and heldout splits")
                    pair_splits[row["pair_id"]] = row["split"]
                    counts[f"{row['corpus']}:{row['split']}"] += 1
                    mutation_counts[row["mutation"]["kind"]] += 1
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        if any(sha256(Path(source["path"])) != source["sha256"] for source in sources):
            raise ValueError("A source file changed during corpus preparation")
        manifest.update(status="complete", output_sha256=sha256(output), output_bytes=output.stat().st_size,
                        records=len(ids), split_counts=dict(counts), mutation_counts=dict(mutation_counts),
                        exact_duplicate_records_skipped=skipped_duplicates, source_groups=len(pair_splits))
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _publish_manifest(manifest_path, manifest)
    return manifest


def ingest_books(output: Path, manifest_path: Path, *, book_dir=None, stable_seconds=30) -> dict:
    """Append each stable local book content hash once; no source-file copying.

    Readers pin complete manifest/hash snapshots. A transient ingesting status or
    hash mismatch means defer new planning; previously queued jobs remain valid.
    One writer owns the manifest through an advisory lock for this whole call.
    """
    import fcntl

    if not math.isfinite(stable_seconds) or stable_seconds < 0:
        raise ValueError("Book stability age cannot be negative")
    # Lock the corpus inode, which is append-only across hot ingestion calls.
    with output.open("r+", encoding="utf-8") as destination:
        fcntl.flock(destination.fileno(), fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("output_sha256") != sha256(output):
            raise ValueError("Hot ingestion requires a complete matching corpus/manifest snapshot")
        directory = Path(book_dir) if book_dir else Path(manifest["book_dir"])
        manifest["book_dir"] = str(directory.resolve())
        seen = {source["sha256"] for source in manifest["sources"] if source["corpus"] == "skippy"}
        checks = manifest.setdefault("book_file_checks", {})
        extractor_sha = sha256(Path(__file__))
        pending, additions, new_sources = [], [], []
        paths = book_candidates(directory)
        for path in paths:
            stat = path.stat()
            stamp = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            key = str(path.resolve())
            previous = checks.get(key, {})
            if time.time() - stat.st_mtime < stable_seconds:
                pending.append({"path": key, "reason": "file still within stability interval"})
                continue
            cached = previous.get("state") in {"ingested", "duplicate"} or (
                previous.get("state") == "invalid" and previous.get("extractor_sha256") == extractor_sha)
            if all(previous.get(k) == v for k, v in stamp.items()) and cached:
                if previous["state"] == "invalid":
                    pending.append({"path": key, "reason": previous["error"]})
                continue
            identity = sha256(path)
            if identity in seen:
                checks[key] = {**stamp, "sha256": identity, "state": "duplicate"}
                continue
            try:
                rows = list(book_records(path, seed=manifest["split_seed"],
                                         max_chars=manifest["selection"]["max_book_chunk_chars"]))
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns) or sha256(path) != identity:
                    raise ValueError("Book changed during extraction; will retry after modification stabilizes")
                if not rows:
                    raise ValueError("Book contains no extractable text")
            except (ValueError, KeyError, OSError, UnicodeError, ET.ParseError, zipfile.BadZipFile) as error:
                checks[key] = {**stamp, "sha256": identity, "state": "invalid", "error": str(error)[:1000],
                               "extractor_sha256": extractor_sha}
                pending.append({"path": key, "reason": str(error)[:1000]})
                continue
            additions.extend(rows)
            new_sources.append(source_info(path, "skippy"))
            seen.add(identity)
            checks[key] = {**stamp, "sha256": identity, "state": "ingested"}
        manifest["pending_books"] = pending
        manifest["books_status"] = "available" if seen else "pending_redownload"
        if not additions:
            _publish_manifest(manifest_path, manifest)
            return {"appended_records": 0, "appended_books": 0, "books_status": manifest["books_status"], "pending_books": pending}
        original_size = output.stat().st_size
        original_manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "ingesting"
        _publish_manifest(manifest_path, manifest)
        try:
            destination.seek(0, os.SEEK_END)
            for row in additions:
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            counts, mutations = Counter(manifest["split_counts"]), Counter(manifest["mutation_counts"])
            for row in additions:
                counts[f"{row['corpus']}:{row['split']}"] += 1
                mutations[row["mutation"]["kind"]] += 1
            manifest.update(status="complete", output_sha256=sha256(output), output_bytes=output.stat().st_size,
                            records=manifest["records"] + len(additions), generation=manifest["generation"] + 1,
                            split_counts=dict(counts), mutation_counts=dict(mutations),
                            source_groups=manifest["source_groups"] + len({row["pair_id"] for row in additions}),
                            sources=manifest["sources"] + new_sources)
            _publish_manifest(manifest_path, manifest)
        except Exception:
            # Only our uncommitted append is rolled back; accepted corpus bytes
            # and book files remain unchanged.
            destination.flush()
            destination.truncate(original_size)
            destination.flush()
            os.fsync(destination.fileno())
            _publish_manifest(manifest_path, original_manifest)
            raise
        return {"appended_records": len(additions), "appended_books": len(new_sources),
                "books_status": manifest["books_status"], "generation": manifest["generation"], "pending_books": pending}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("prepare", "ingest", "watch"), default="prepare")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--book", type=Path, action="append", default=[], help="Explicit local Skippy EPUB/TXT file; may be repeated; no recursive discovery")
    parser.add_argument("--book-dir", type=Path, help="Local drop directory; default root/books; recognized Skippy child folders one level deep are supported; Downloads excludes unrelated files")
    parser.add_argument("--output", type=Path, help="Defaults to root/data/persona_manager_corpus.jsonl")
    parser.add_argument("--manifest", type=Path, help="Defaults to root/data/persona_manager_corpus_manifest.json")
    parser.add_argument("--seed", type=int, default=7035)
    parser.add_argument("--chunk-chars", type=int, default=1600)
    parser.add_argument("--max-prompt-chars", type=int, default=6000)
    parser.add_argument("--contrastive-limit", type=int, help="Optional explicit source-line limit; default scans the complete local JSONL")
    parser.add_argument("--include-contrastive", action="store_true", help="Also include the existing local contrastive prompt bank; default is SCOTUS+BigFive")
    parser.add_argument("--stable-seconds", type=float, default=30, help="Minimum unchanged-file age before EPUB/TXT ingestion")
    parser.add_argument("--interval", type=float, default=30, help="Watch poll interval in seconds (minimum 5)")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate these owned outputs in place")
    args = parser.parse_args()
    output = args.output or args.root / "data/persona_manager_corpus.jsonl"
    manifest = args.manifest or args.root / "data/persona_manager_corpus_manifest.json"
    if args.command == "prepare":
        result = prepare(args.root, output, manifest, books=args.book,
            seed=args.seed, max_chars=args.chunk_chars, contrastive_limit=args.contrastive_limit,
            max_prompt_chars=args.max_prompt_chars, overwrite=args.overwrite,
            include_contrastive=args.include_contrastive, book_dir=args.book_dir)
        print(json.dumps({key: result[key] for key in ("status", "records", "split_counts", "books_status", "output_sha256")}))
    else:
        if args.interval < 5:
            raise ValueError("Book watch interval must be at least five seconds")
        previous = None
        while True:
            result = ingest_books(output, manifest, book_dir=args.book_dir, stable_seconds=args.stable_seconds)
            if result != previous:
                print(json.dumps(result), flush=True)
            previous = result
            if args.command != "watch":
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
