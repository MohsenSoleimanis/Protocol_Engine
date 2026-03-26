"""
Block Renderer — Converts blocks to markdown with metadata tags.

LEGACY: Used by ProtocolStore.get_pages() for /api/section endpoint.
Orchestrator now uses _fetch_raw_pages() in orchestrator.py which
sends raw PyMuPDF text + merged parsed tables directly to the LLM.

The standard format for sending structured content to LLMs:
  - Markdown for content (LLMs trained on billions of MD files)
  - XML tags for metadata (page, section, type for grounding)

Each block becomes:
  <block page="41" section="5.2" type="numbered_item">
  1. History of allergy to any component of the vaccine
  </block>
"""
from __future__ import annotations


def render_blocks_for_llm(blocks: list[dict]) -> str:
    """
    Render a list of blocks as markdown with metadata tags.
    Blocks should be pre-sorted by (page, position).
    """
    parts = []
    current_section = ""
    
    for block in blocks:
        section = block.get("section_id", "")
        page = block.get("page", 0)
        btype = block.get("type", "paragraph")
        text = block.get("text", "")
        
        if not text.strip():
            continue
        
        # Section header when section changes
        if section != current_section and section:
            current_section = section
        
        # Render based on type
        rendered = _render_block(block)
        
        # Wrap in metadata tag
        page_str = str(page)
        if block.get("page_range"):
            pr = block["page_range"]
            if len(pr) > 1:
                page_str = f"{pr[0]}-{pr[-1]}"
        
        parts.append(
            f'<block page="{page_str}" section="{section}" type="{btype}">\n'
            f'{rendered}\n'
            f'</block>'
        )
    
    return "\n\n".join(parts)


def _render_block(block: dict) -> str:
    """Render a single block as markdown."""
    btype = block.get("type", "paragraph")
    text = block.get("text", "")
    
    if btype == "section_heading":
        return f"## {text}"
    
    elif btype == "bold_heading":
        return f"**{text}**"
    
    elif btype == "numbered_item":
        # Clean up: ensure it starts with number
        return text
    
    elif btype == "bullet_item":
        # Ensure markdown bullet format
        clean = text.lstrip("•-○– \t")
        return f"- {clean}"
    
    elif btype == "table":
        # Table text is already markdown from block_index
        caption = block.get("caption", "")
        if caption and not text.startswith(f"**{caption}"):
            return f"### {caption}\n\n{text}"
        return text
    
    elif btype == "footnote":
        marker = block.get("marker", "")
        if marker:
            return f"> ^{marker}: {text}"
        return f"> {text}"
    
    elif btype == "definition":
        return f"**{text}**"
    
    elif btype == "referenced_definition":
        ref_source = block.get("ref_source", "")
        return f"**Referenced: {ref_source}**\n{text}"
    
    else:
        return text


def render_context_bundle(
    primary_blocks: list[dict],
    ref_blocks: list[dict] = None,
    footnote_blocks: list[dict] = None,
) -> str:
    """
    Render a context bundle: primary content + references + footnotes.
    This is what gets sent to the extraction LLM.
    """
    parts = []
    
    # Primary content
    if primary_blocks:
        primary_md = render_blocks_for_llm(primary_blocks)
        parts.append(primary_md)
    
    # Referenced content (cross-references resolved)
    if ref_blocks:
        parts.append("\n--- REFERENCED CONTENT ---\n")
        ref_md = render_blocks_for_llm(ref_blocks)
        parts.append(ref_md)
    
    # Footnotes
    if footnote_blocks:
        parts.append("\n--- FOOTNOTES ---\n")
        fn_md = render_blocks_for_llm(footnote_blocks)
        parts.append(fn_md)
    
    return "\n\n".join(parts)
