# Drawing plan

- Canvas: 504 × 218 pt, white background, double-column systems architecture figure.
- Grid: three horizontal bands from top to bottom: Online turn path (y=8–76), Write path (y=86–151), Compact/read path (y=161–210). Keep 8 pt gutters between bands.
- Main spine: User turn → Tokenizer → Turn controller + capacity gate → Native KV tail → Reader-equipped decoder → Assistant tokens, centered at y=43. Use 1.1 pt dark arrows and step badges ①–③, then ⑥ on the reader-state return.
- Write branch: Turn controller at x≈153 sends a vertical control arrow to Write timing at y≈104; Write timing → Sigma writer → Written memory H runs left-to-right in the middle band. Assistant tokens send a thin dashed scheduling edge to Write timing. Label the policy choices inside the Write timing box as `turn | every n tokens | segment`.
- Compact branch: Written memory H sends a dashed conditional edge labelled `on overflow / first compact` to Compaction at x≈414. Compaction → Readable memory H → Reader state occupies the lower band, with the final edge routed upward into the Reader-equipped decoder. A thin solid edge from Compaction to Native KV tail is labelled `rebuild recent tail`.
- Containers: each band is a pale flat rectangle with a small bold title in the upper-left gutter. Use solid light outlines; reserve dashed lines for the conditional compact path only.
- Palette: blue-orange. Blue modules for existing caches and model components, pale yellow for writer scheduling, pale green for compact/read state, orange accent for the harness controller. Near-black text, 0.55 pt box strokes, 1.1 pt main arrows, 0.75 pt secondary arrows.
- Text: Helvetica/TeX Gyre Heros; lane titles 8 pt bold, node labels 6.7 pt bold, sublabels 6.1 pt. Keep all labels inside boxes, with no text below 6 pt.
- Final checks: no edge crosses a label; no duplicate component labels; the first-compact gate is readable in five seconds; dashed style has one meaning; all six step badges follow the main path; the compact/read path visibly returns to the decoder and the native tail is rebuilt.
