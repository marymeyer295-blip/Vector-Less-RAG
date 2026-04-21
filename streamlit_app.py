import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st
from dotenv import load_dotenv

import litellm

from pageindex.utils import ConfigLoader, extract_json


load_dotenv()


@dataclass(frozen=True)
class DocNode:
    node_id: str
    title: str
    start_index: Optional[int]
    end_index: Optional[int]
    summary: str
    path_titles: tuple[str, ...]

    @property
    def page_range(self) -> str:
        if self.start_index is None and self.end_index is None:
            return ""
        if self.start_index is None:
            return f"{self.end_index}"
        if self.end_index is None or self.end_index == self.start_index:
            return f"{self.start_index}"
        return f"{self.start_index}-{self.end_index}"

    @property
    def path(self) -> str:
        return " > ".join(self.path_titles)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent


def _results_dir() -> Path:
    return _workspace_root() / "results"


def _examples_docs_dir() -> Path:
    return _workspace_root() / "examples" / "documents"


def _list_structure_json_files() -> list[Path]:
    results = _results_dir()
    if not results.exists():
        return []
    return sorted(results.glob("*_structure.json"), key=lambda p: p.name.lower())


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_structure(structure: list[dict], parents: tuple[str, ...] = ()) -> list[DocNode]:
    nodes: list[DocNode] = []
    for item in structure or []:
        title = str(item.get("title", "")).strip()
        node_id = str(item.get("node_id", "")).strip()
        start_index = item.get("start_index")
        end_index = item.get("end_index")
        summary = str(item.get("summary", "")).strip()
        path_titles = parents + ((title or node_id or "Untitled"),)
        nodes.append(
            DocNode(
                node_id=node_id,
                title=title,
                start_index=start_index if isinstance(start_index, int) else None,
                end_index=end_index if isinstance(end_index, int) else None,
                summary=summary,
                path_titles=path_titles,
            )
        )
        if isinstance(item.get("nodes"), list):
            nodes.extend(_flatten_structure(item["nodes"], parents=path_titles))
    return nodes


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+-]{1,}")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text or "")}


def _lexical_rank(query: str, nodes: Iterable[DocNode], k: int = 30) -> list[DocNode]:
    q = _tokenize(query)
    if not q:
        return list(nodes)[:k]

    scored: list[tuple[float, DocNode]] = []
    for node in nodes:
        hay = f"{node.title}\n{node.path}\n{node.summary}"
        t = _tokenize(hay)
        overlap = len(q & t)
        denom = max(1, len(q))
        score = overlap / denom
        if score > 0:
            scored.append((score, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:k]]


def _guess_pdf_path(doc_name: str) -> Optional[Path]:
    if not doc_name:
        return None
    candidates = [
        _workspace_root() / doc_name,
        _results_dir() / doc_name,
        _examples_docs_dir() / doc_name,
    ]
    for p in candidates:
        if p.exists() and p.suffix.lower() == ".pdf":
            return p
    # fallback: search examples/docs by basename
    base = Path(doc_name).name
    p = _examples_docs_dir() / base
    if p.exists() and p.suffix.lower() == ".pdf":
        return p
    return None


def _extract_pages(pdf_path: Path, pages: list[int], max_chars_per_page: int = 6000) -> list[dict]:
    # Prefer PyMuPDF for robustness; fall back to PyPDF2.
    pages = sorted({p for p in pages if isinstance(p, int) and p >= 1})
    if not pages:
        return []
    try:
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(pdf_path))
        except Exception:
            import pymupdf  # type: ignore
            doc = pymupdf.open(str(pdf_path))
        out = []
        page_count = getattr(doc, "page_count", None) or getattr(doc, "pageCount", None) or 0
        for p in pages:
            if 1 <= p <= int(page_count):
                page = doc.load_page(p - 1) if hasattr(doc, "load_page") else doc[p - 1]
                text = page.get_text() if hasattr(page, "get_text") else (page.getText() or "")
                out.append({"page": p, "content": text[:max_chars_per_page]})
        doc.close()
        return out
    except Exception:
        pass

    try:
        import PyPDF2  # type: ignore

        out = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            for p in pages:
                if 1 <= p <= total:
                    text = reader.pages[p - 1].extract_text() or ""
                    out.append({"page": p, "content": text[:max_chars_per_page]})
        return out
    except Exception:
        return []


def _pick_pages_from_nodes(nodes: list[DocNode], max_pages: int = 8) -> list[int]:
    pages: list[int] = []
    for node in nodes:
        if node.start_index is None:
            continue
        end = node.end_index if node.end_index is not None else node.start_index
        # Guard: avoid huge ranges
        for p in range(node.start_index, min(end, node.start_index + 5) + 1):
            pages.append(p)
            if len(set(pages)) >= max_pages:
                return sorted(set(pages))[:max_pages]
    return sorted(set(pages))[:max_pages]


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


def _llm_model_default() -> str:
    try:
        opt = ConfigLoader().load()
        return str(getattr(opt, "retrieve_model", None) or getattr(opt, "model", "") or "")
    except Exception:
        return ""


def _llm_chat(model: str, messages: list[dict], temperature: float = 0) -> str:
    if model:
        model = model.removeprefix("litellm/")
    resp = litellm.completion(model=model, messages=messages, temperature=temperature)
    return resp.choices[0].message.content or ""


def _rerank_with_llm(model: str, question: str, candidates: list[DocNode], k: int = 6) -> list[DocNode]:
    if not candidates:
        return []
    # Keep prompt small: titles + page ranges + short summaries.
    items = []
    for n in candidates[:30]:
        summ = (n.summary or "").strip().replace("\n", " ")
        if len(summ) > 260:
            summ = summ[:260] + "…"
        items.append(
            {
                "node_id": n.node_id,
                "title": n.title,
                "path": n.path,
                "pages": n.page_range,
                "summary": summ,
            }
        )
    prompt = (
        "Pick the most relevant sections for answering the user question.\n"
        "Return JSON ONLY, with this schema:\n"
        '{ "top_node_ids": ["0001", "0023", ...] }\n\n'
        f"Question: {question}\n\n"
        f"Candidates: {json.dumps(items, ensure_ascii=False)}"
    )
    content = _llm_chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a retrieval router. Select relevant node_ids only. No extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    data = extract_json(content)
    ids = data.get("top_node_ids") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        return candidates[:k]
    wanted = [str(x).zfill(4) if str(x).isdigit() else str(x) for x in ids]
    by_id = {n.node_id: n for n in candidates}
    out = [by_id[i] for i in wanted if i in by_id]
    return (out + [n for n in candidates if n not in out])[:k]


def _answer_with_llm(
    model: str,
    question: str,
    chosen_nodes: list[DocNode],
    page_text: list[dict],
    history: list[dict],
) -> str:
    sources = []
    for n in chosen_nodes[:6]:
        sources.append(
            {
                "node_id": n.node_id,
                "title": n.title,
                "path": n.path,
                "pages": n.page_range,
                "summary": (n.summary or "").strip(),
            }
        )

    excerpts = []
    for p in page_text[:12]:
        t = (p.get("content") or "").strip()
        if len(t) > 4500:
            t = t[:4500] + "…"
        excerpts.append({"page": p.get("page"), "text": t})

    user_payload = {
        "question": question,
        "sources": sources,
        "excerpts": excerpts,
        "constraints": {
            "answer_only_from_sources_and_excerpts": True,
            "cite_pages_and_node_ids": True,
        },
    }

    sys = (
        "You are a document QA assistant.\n"
        "Answer ONLY using the provided 'sources' (summaries) and 'excerpts' (page text).\n"
        "If the answer is not supported, say what is missing.\n"
        "Cite sources inline using: (p. X, node_id ####).\n"
    )

    # Keep a small history for conversational continuity.
    trimmed = []
    for msg in history[-6:]:
        if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
            trimmed.append({"role": msg["role"], "content": msg["content"][:1200]})

    messages = [{"role": "system", "content": sys}] + trimmed + [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}]
    return _llm_chat(model=model, messages=messages, temperature=0)


def _answer_without_llm(question: str, chosen_nodes: list[DocNode]) -> str:
    lines = [
        "No LLM API key detected, so I can’t generate a natural-language answer.",
        "Relevant sections from your JSON tree:",
    ]
    for n in chosen_nodes[:8]:
        pr = f"p. {n.page_range}" if n.page_range else "pages: n/a"
        lines.append(f"- {n.path} ({pr}, node_id {n.node_id})")
    lines.append("")
    lines.append("Set `OPENAI_API_KEY` (or Azure/LiteLLM env vars) and reload to enable chat answering.")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="PageIndex Simple Chat", layout="wide")

    st.title("PageIndex — Simple Chat UI (local)")
    st.caption("Loads a `*_structure.json` tree and answers questions by routing to relevant sections + PDF pages.")

    json_files = _list_structure_json_files()
    if not json_files:
        st.error("No `*_structure.json` files found in `PageIndex/results/`.")
        st.stop()

    with st.sidebar:
        st.header("Document")
        chosen = st.selectbox(
            "Structure JSON",
            options=json_files,
            format_func=lambda p: p.name,
        )
        doc = _read_json(Path(chosen))
        doc_name = str(doc.get("doc_name", "") or "")
        structure = doc.get("structure") or []
        nodes = _flatten_structure(structure)

        st.write(f"Nodes: `{len(nodes)}`")
        st.write(f"Doc: `{doc_name or 'unknown'}`")

        st.header("Model")
        default_model = _llm_model_default()
        model = st.text_input("LLM model (LiteLLM)", value=default_model or "gpt-4o-mini")
        st.caption("Uses LiteLLM; set `.env` with your provider keys.")

        st.header("PDF")
        guessed = _guess_pdf_path(doc_name)
        pdf_path_str = st.text_input("PDF path (optional)", value=str(guessed) if guessed else "")
        uploaded = st.file_uploader("Or upload PDF", type=["pdf"])

        st.header("Retrieval")
        max_pages = st.slider("Max pages to fetch", 1, 12, 8)
        use_llm_rerank = st.checkbox("LLM rerank sections", value=True)

        st.divider()
        st.write("LLM available:", "✅" if _llm_available() else "❌")

    pdf_path: Optional[Path] = None
    tmp_pdf: Optional[Path] = None
    if uploaded is not None:
        tmp_dir = _workspace_root() / ".tmp_ui"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_pdf = tmp_dir / uploaded.name
        tmp_pdf.write_bytes(uploaded.getbuffer())
        pdf_path = tmp_pdf
    elif pdf_path_str.strip():
        p = Path(pdf_path_str.strip())
        if p.exists():
            pdf_path = p

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")

    question = st.chat_input("Ask a question about the document…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieval
    lexical = _lexical_rank(question, nodes, k=30)
    chosen_nodes = lexical[:6]
    if _llm_available() and use_llm_rerank:
        try:
            chosen_nodes = _rerank_with_llm(model=model, question=question, candidates=lexical, k=6)
        except Exception:
            chosen_nodes = lexical[:6]

    pages = _pick_pages_from_nodes(chosen_nodes, max_pages=max_pages)
    page_text = _extract_pages(pdf_path, pages) if pdf_path else []

    # Answer
    with st.chat_message("assistant"):
        if _llm_available():
            try:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                answer = _answer_with_llm(model=model, question=question, chosen_nodes=chosen_nodes, page_text=page_text, history=history)
            except Exception as e:
                answer = f"Error calling the LLM: {e}"
        else:
            answer = _answer_without_llm(question, chosen_nodes)

        st.markdown(answer)

        source_lines = []
        for n in chosen_nodes[:8]:
            pr = f"p. {n.page_range}" if n.page_range else "pages: n/a"
            source_lines.append(f"{n.path} ({pr}, node_id {n.node_id})")
        with st.expander("Sources"):
            for s in source_lines:
                st.markdown(f"- `{s}`")
            if pdf_path and page_text:
                st.markdown("---")
                st.markdown(f"PDF: `{str(pdf_path)}`")
                st.markdown("Pages fetched:")
                st.markdown(f"`{', '.join(str(p['page']) for p in page_text if p.get('page'))}`")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": source_lines})


if __name__ == "__main__":
    main()
