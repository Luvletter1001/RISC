# OV-CapFlow Three-Figure Paper Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate three top-conference-quality OV-CapFlow methodology figures as editable draw.io sources plus reproducible SVG, PDF, and PNG exports.

**Architecture:** A declarative Python figure specification is the single source of truth for dimensions, semantic roles, nodes, edges, captions, and evidence status. One renderer converts that specification to Matplotlib vector/raster outputs, while a second writer converts the same specification to editable diagrams.net XML; tests enforce the scientific boundary between validated T7 modules and candidate OV-CapFlow mechanisms.

**Tech Stack:** Python 3.8, standard-library dataclasses/XML, Matplotlib, Pillow, pytest, diagrams.net XML, SVG/PDF/PNG.

---

## File map

| path | responsibility |
|---|---|
| `projects/OVCapFlow/tools/paper_figure_spec.py` | shared dataclasses, colour/font tokens, exact node/edge content for all three figures |
| `projects/OVCapFlow/tools/generate_paper_figures.py` | Matplotlib renderer, draw.io XML writer, CLI, output audit |
| `tests/test_projects/ov_capflow/test_paper_figures.py` | scientific-boundary, layout, XML, vector, and export regression tests |
| `docs/figures/ov_capflow/README.md` | figure descriptions, captions, evidence convention, regeneration command |
| `docs/figures/ov_capflow/ov_capflow_t7_pipeline.*` | editable source and exports for the validated T7 pipeline |
| `docs/figures/ov_capflow/ov_capflow_full_architecture.*` | editable source and exports for the complete architecture |
| `docs/figures/ov_capflow/ov_capflow_semantic_capacity_flow.*` | editable source and exports for the mechanism zoom-in |

The generator owns all binary/text exports. Authors edit the declarative spec
or the generated draw.io file, but any camera-ready change must be reflected
back into the declarative spec before final export.

### Task 1: Establish the declarative figure contract

**Files:**
- Create: `projects/OVCapFlow/tools/paper_figure_spec.py`
- Create: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Write the failing contract tests**

```python
from projects.OVCapFlow.tools.paper_figure_spec import (
    FIGURE_SPECS, PALETTE, FigureSpec)


def test_three_specs_have_publication_dimensions():
    assert set(FIGURE_SPECS) == {
        'ov_capflow_t7_pipeline',
        'ov_capflow_full_architecture',
        'ov_capflow_semantic_capacity_flow',
    }
    for spec in FIGURE_SPECS.values():
        assert isinstance(spec, FigureSpec)
        assert spec.width_mm == 178.0
        assert 54.0 <= spec.height_mm <= 68.0
        assert spec.min_font_pt >= 8.0
        for node in spec.nodes:
            assert node.font_pt >= spec.min_font_pt
            assert 0.0 <= node.x < node.x + node.width <= spec.width_mm
            assert 0.0 <= node.y < node.y + node.height <= spec.height_mm


def test_palette_is_colour_blind_safe_and_role_complete():
    assert set(PALETTE) == {
        'visual', 'text', 'query', 'method', 'output', 'training',
        'neutral', 'ink', 'paper'
    }
    assert PALETTE['visual'].stroke == '#0072B2'
    assert PALETTE['text'].stroke == '#009E73'
    assert PALETTE['method'].stroke == '#E69F00'


def test_all_node_ids_are_unique_and_edges_resolve():
    for spec in FIGURE_SPECS.values():
        node_ids = [node.id for node in spec.nodes]
        assert len(node_ids) == len(set(node_ids))
        known = set(node_ids)
        for edge in spec.edges:
            assert edge.source in known
            assert edge.target in known
```

- [ ] **Step 2: Run the tests and verify collection fails**

Run:

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_paper_figures.py -q
```

Expected: FAIL during import with `ModuleNotFoundError` for
`paper_figure_spec`.

- [ ] **Step 3: Implement the shared dataclasses and validation**

Create `paper_figure_spec.py` with these public types and tokens:

```python
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ColourToken:
    stroke: str
    fill: str


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    role: str
    kind: str = 'module'
    font_pt: float = 8.5
    dashed: bool = False
    italic: bool = False


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    label: str = ''
    style: str = 'solid'


@dataclass(frozen=True)
class FigureSpec:
    slug: str
    width_mm: float
    height_mm: float
    min_font_pt: float
    nodes: Tuple[Node, ...]
    edges: Tuple[Edge, ...]
    caption: str

    def validate(self) -> None:
        if self.width_mm != 178.0:
            raise ValueError('paper figures must be 178 mm wide')
        if not 54.0 <= self.height_mm <= 68.0:
            raise ValueError('paper figure height is outside the design spec')
        if self.min_font_pt < 8.0:
            raise ValueError('post-scaling font must be at least 8 pt')
        for node in self.nodes:
            if node.font_pt < self.min_font_pt:
                raise ValueError('node font is smaller than the figure floor')
            if not 0.0 <= node.x < node.x + node.width <= self.width_mm:
                raise ValueError('node exceeds the horizontal canvas')
            if not 0.0 <= node.y < node.y + node.height <= self.height_mm:
                raise ValueError('node exceeds the vertical canvas')
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError('figure node ids must be unique')
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError('edge endpoint is missing from the figure')


PALETTE: Dict[str, ColourToken] = {
    'visual': ColourToken('#0072B2', '#E7F4FA'),
    'text': ColourToken('#009E73', '#E7F7F2'),
    'query': ColourToken('#7A5195', '#F0EAF6'),
    'method': ColourToken('#E69F00', '#FFF4D6'),
    'output': ColourToken('#D55E00', '#FCEAE4'),
    'training': ColourToken('#7A828B', '#F1F3F5'),
    'neutral': ColourToken('#9AA4AE', '#F7F8FA'),
    'ink': ColourToken('#263442', '#FFFFFF'),
    'paper': ColourToken('#FFFFFF', '#FFFFFF'),
}
```

Define temporary empty tuples for each of the three `FigureSpec` instances and
export them through `FIGURE_SPECS`; Tasks 2-4 replace those empty contents.

- [ ] **Step 4: Run the contract tests**

Run the command from Step 2.

Expected: `3 passed`.

- [ ] **Step 5: Commit the shared contract**

```bash
rtk git add projects/OVCapFlow/tools/paper_figure_spec.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "test: define paper figure contract"
```

### Task 2: Specify Figure 1, the validated T7 pipeline

**Files:**
- Modify: `projects/OVCapFlow/tools/paper_figure_spec.py`
- Modify: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Add the evidence-boundary test for Figure 1**

```python
def test_t7_figure_contains_only_active_inference_modules():
    spec = FIGURE_SPECS['ov_capflow_t7_pipeline']
    labels = {node.label for node in spec.nodes}
    assert {
        'Aerial Image', 'Category Prompts', 'Swin-T', 'BERT',
        'Cross-modal Encoder', 'Fixed Content Queries\nQ = 600',
        'Learned 5-D\nRotated References', 'Rotated Decoder x6',
        'Text Similarity', 'Direct 5-D RBox',
        '600 Ordered Predictions'
    } <= labels
    forbidden = {'Capacity Head', 'Scene Density Head', 'Null Reservoir'}
    assert labels.isdisjoint(forbidden)
    candidate_nodes = [node for node in spec.nodes if node.dashed]
    assert candidate_nodes == []


def test_t7_training_path_is_dotted_and_does_not_feed_readout():
    spec = FIGURE_SPECS['ov_capflow_t7_pipeline']
    training_ids = {
        node.id for node in spec.nodes if node.role == 'training'
    }
    assert training_ids == {'train_groups', 'dn_queries', 'o2o_losses'}
    training_edges = [edge for edge in spec.edges if edge.style == 'dotted']
    assert training_edges
    assert all(edge.target == 'decoder' for edge in training_edges)
```

- [ ] **Step 2: Run the two tests and verify failure**

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_paper_figures.py \
  -q -k 't7_figure or t7_training'
```

Expected: FAIL because the T7 spec has no nodes.

- [ ] **Step 3: Implement the exact T7 node and edge specification**

Add a `build_t7_pipeline()` function. Use millimetres as coordinates. The
main modules occupy `y=13..42`; the training lane occupies `y=47..59`.

```python
def build_t7_pipeline() -> FigureSpec:
    nodes = (
        Node('image', 'Aerial Image', 3, 16, 15, 17, 'visual', 'aerial'),
        Node('prompts', 'Category Prompts', 3, 36, 15, 8, 'text', 'prompt'),
        Node('swin', 'Swin-T', 25, 14, 17, 10, 'visual'),
        Node('bert', 'BERT', 25, 30, 17, 10, 'text'),
        Node('encoder', 'Cross-modal Encoder', 48, 20, 24, 14, 'visual'),
        Node('queries', 'Fixed Content Queries\nQ = 600',
             78, 12, 22, 12, 'query'),
        Node('references', 'Learned 5-D\nRotated References',
             78, 29, 22, 12, 'query'),
        Node('decoder', 'Rotated Decoder x6', 107, 18, 24, 16, 'visual'),
        Node('text_readout', 'Text Similarity', 137, 13, 17, 10, 'text'),
        Node('box_readout', 'Direct 5-D RBox', 137, 28, 17, 10, 'output'),
        Node('predictions', '600 Ordered Predictions',
             159, 17, 16, 19, 'output', 'rbox_output'),
        Node('train_groups', '3 Shared Q600 Groups',
             72, 49, 27, 8, 'training', font_pt=8.0),
        Node('dn_queries', 'DN Queries',
             102, 49, 18, 8, 'training', font_pt=8.0),
        Node('o2o_losses', 'Independent Hungarian O2O Losses',
             123, 49, 35, 8, 'training', font_pt=8.0),
        Node('contract', 'No proposal top-k  |  No global top-k  |  No NMS',
             120, 58.5, 55, 4.5, 'ink', 'text', font_pt=8.0),
    )
    edges = (
        Edge('e_image_swin', 'image', 'swin'),
        Edge('e_prompt_bert', 'prompts', 'bert'),
        Edge('e_swin_encoder', 'swin', 'encoder'),
        Edge('e_bert_encoder', 'bert', 'encoder'),
        Edge('e_encoder_decoder', 'encoder', 'decoder', 'visual-text memory'),
        Edge('e_queries_decoder', 'queries', 'decoder'),
        Edge('e_refs_decoder', 'references', 'decoder'),
        Edge('e_decoder_text', 'decoder', 'text_readout'),
        Edge('e_decoder_box', 'decoder', 'box_readout'),
        Edge('e_text_predictions', 'text_readout', 'predictions'),
        Edge('e_box_predictions', 'box_readout', 'predictions'),
        Edge('e_groups_decoder', 'train_groups', 'decoder', style='dotted'),
        Edge('e_dn_decoder', 'dn_queries', 'decoder', style='dotted'),
        Edge('e_o2o_decoder', 'o2o_losses', 'decoder', style='dotted'),
    )
    return FigureSpec(
        slug='ov_capflow_t7_pipeline', width_mm=178.0, height_mm=64.0,
        min_font_pt=8.0, nodes=nodes, edges=edges,
        caption=T7_CAPTION)
```

Set `T7_CAPTION` to the approved caption in Section 4.3 of the design spec.

- [ ] **Step 4: Run all paper-figure tests**

Run the Task 1 Step 2 command.

Expected: all tests PASS.

- [ ] **Step 5: Commit Figure 1 content**

```bash
rtk git add projects/OVCapFlow/tools/paper_figure_spec.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "feat: specify validated T7 pipeline figure"
```

### Task 3: Specify Figure 2, the complete architecture

**Files:**
- Modify: `projects/OVCapFlow/tools/paper_figure_spec.py`
- Modify: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Add candidate-status regression tests**

```python
def test_full_architecture_marks_every_candidate_as_dashed():
    spec = FIGURE_SPECS['ov_capflow_full_architecture']
    nodes = {node.label: node for node in spec.nodes}
    candidates = {
        'Capacity Head c_i', 'Scene Density Head N_hat',
        'Parent-preserving\nSemantic Residual', 'Null Reservoir'
    }
    assert candidates <= set(nodes)
    for label in candidates:
        assert nodes[label].role == 'method'
        assert nodes[label].dashed is True


def test_full_architecture_never_prunes_queries():
    spec = FIGURE_SPECS['ov_capflow_full_architecture']
    text = ' '.join(node.label.lower() for node in spec.nodes)
    assert 'all q600 rows retained' in text
    for forbidden in ('top-k', 'nms', 'prune', 'survivor'):
        assert forbidden not in text
```

- [ ] **Step 2: Run and verify the candidate tests fail**

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_paper_figures.py \
  -q -k 'full_architecture'
```

Expected: FAIL because Figure 2 is empty.

- [ ] **Step 3: Implement the complete architecture specification**

Build three zones with these exact nodes:

```python
def build_full_architecture() -> FigureSpec:
    nodes = (
        Node('memory', 'Visual-Text Memory', 4, 18, 24, 14, 'visual'),
        Node('native', 'Immutable Native Queries', 4, 37, 24, 14, 'query'),
        Node('parent', 'Parent Decoder', 42, 14, 28, 13, 'visual'),
        Node('transported', 'Transported Evidence q_parent',
             42, 32, 28, 11, 'visual'),
        Node('capacity', 'Capacity Head c_i',
             79, 12, 24, 12, 'method', dashed=True),
        Node('density', 'Scene Density Head N_hat',
             79, 30, 24, 12, 'method', dashed=True),
        Node('residual', 'Parent-preserving\nSemantic Residual',
             109, 19, 29, 19, 'method', dashed=True),
        Node('repeat', 'Semantic-Capacity Flow Layer x6',
             38, 7, 104, 43, 'neutral', 'container', font_pt=9.5),
        Node('null', 'Null Reservoir',
             147, 13, 25, 12, 'method', dashed=True),
        Node('readout', 'Per-query Class + 5-D RBox',
             147, 31, 25, 12, 'output'),
        Node('all_rows', 'All Q600 Rows Retained',
             147, 48, 25, 10, 'output'),
        Node('capacity_loss', 'Capacity Mass',
             75, 57, 22, 8, 'training', font_pt=8.0),
        Node('density_loss', 'Density Count',
             100, 57, 22, 8, 'training', font_pt=8.0),
        Node('null_loss', 'Null Calibration',
             125, 57, 22, 8, 'training', font_pt=8.0),
    )
    edges = (
        Edge('e_memory_parent', 'memory', 'parent'),
        Edge('e_parent_transport', 'parent', 'transported'),
        Edge('e_transport_capacity', 'transported', 'capacity'),
        Edge('e_memory_density', 'memory', 'density'),
        Edge('e_native_residual', 'native', 'residual'),
        Edge('e_transport_residual', 'transported', 'residual'),
        Edge('e_capacity_residual', 'capacity', 'residual'),
        Edge('e_residual_null', 'residual', 'null'),
        Edge('e_null_readout', 'null', 'readout'),
        Edge('e_readout_rows', 'readout', 'all_rows'),
        Edge('e_capacity_loss', 'capacity_loss', 'capacity', style='dotted'),
        Edge('e_density_loss', 'density_loss', 'density', style='dotted'),
        Edge('e_null_loss', 'null_loss', 'null', style='dotted'),
    )
    return FigureSpec(
        slug='ov_capflow_full_architecture', width_mm=178.0,
        height_mm=68.0, min_font_pt=8.0, nodes=nodes, edges=edges,
        caption=FULL_ARCHITECTURE_CAPTION)
```

Render `repeat` behind its child nodes. Set `FULL_ARCHITECTURE_CAPTION` to the
approved caption in Section 5.3 of the design spec.

- [ ] **Step 4: Run all paper-figure tests**

Expected: all tests PASS.

- [ ] **Step 5: Commit Figure 2 content**

```bash
rtk git add projects/OVCapFlow/tools/paper_figure_spec.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "feat: specify complete OV-CapFlow architecture figure"
```

### Task 4: Specify Figure 3, the mechanism zoom-in

**Files:**
- Modify: `projects/OVCapFlow/tools/paper_figure_spec.py`
- Modify: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Add equation and guarantee tests**

```python
def test_semantic_capacity_figure_matches_implemented_equation():
    spec = FIGURE_SPECS['ov_capflow_semantic_capacity_flow']
    labels = {node.label for node in spec.nodes}
    assert (
        'q_out = q_parent + c_i tanh(G[q_native, q_parent]) '
        'P(q_native - q_parent)'
    ) in labels
    assert 'G = 0  =>  q_out = q_parent' in labels
    assert 'c_i = 0  =>  q_out = q_parent' in labels
    assert all('null' not in label.lower() for label in labels)


def test_semantic_capacity_figure_keeps_all_queries():
    spec = FIGURE_SPECS['ov_capflow_semantic_capacity_flow']
    labels = {node.label for node in spec.nodes}
    assert 'Continuous weighting; all Q retained' in labels
```

- [ ] **Step 2: Run and verify the mechanism tests fail**

Run pytest with `-k semantic_capacity_figure`.

Expected: FAIL because Figure 3 is empty.

- [ ] **Step 3: Implement the three-panel mechanism specification**

```python
def build_semantic_capacity_flow() -> FigureSpec:
    nodes = (
        Node('panel_a', '(a) State Separation',
             2, 4, 50, 49, 'neutral', 'container', font_pt=9.5),
        Node('native', 'q_native\nPersistent Identity',
             8, 15, 17, 13, 'query', italic=True),
        Node('parent', 'q_parent\nTransported Evidence',
             30, 15, 17, 13, 'visual', italic=True),
        Node('difference', 'q_native - q_parent',
             17, 35, 22, 10, 'method', dashed=True, italic=True),
        Node('panel_b', '(b) Capacity-controlled Residual',
             56, 4, 76, 49, 'neutral', 'container', font_pt=9.5),
        Node('equation',
             'q_out = q_parent + c_i tanh(G[q_native, q_parent]) '
             'P(q_native - q_parent)',
             60, 14, 68, 13, 'ink', 'math', font_pt=9.0, italic=True),
        Node('capacity', 'Capacity c_i', 63, 35, 18, 10,
             'method', dashed=True),
        Node('gate', 'Zero-init Gate G', 85, 35, 20, 10,
             'method', dashed=True),
        Node('adapter', 'Adapter P', 109, 35, 17, 10,
             'method', dashed=True),
        Node('panel_c', '(c) Exact Guarantees',
             136, 4, 40, 49, 'neutral', 'container', font_pt=9.5),
        Node('zero_gate', 'G = 0  =>  q_out = q_parent',
             141, 14, 30, 9, 'text', 'text', font_pt=8.0),
        Node('zero_capacity', 'c_i = 0  =>  q_out = q_parent',
             141, 28, 30, 9, 'text', 'text', font_pt=8.0),
        Node('all_queries', 'Continuous weighting; all Q retained',
             141, 42, 30, 9, 'training', 'text', font_pt=8.0),
    )
    edges = (
        Edge('e_native_difference', 'native', 'difference'),
        Edge('e_parent_difference', 'parent', 'difference'),
        Edge('e_difference_equation', 'difference', 'equation'),
        Edge('e_capacity_equation', 'capacity', 'equation'),
        Edge('e_gate_equation', 'gate', 'equation'),
        Edge('e_adapter_equation', 'adapter', 'equation'),
        Edge('e_equation_zero_gate', 'equation', 'zero_gate'),
    )
    return FigureSpec(
        slug='ov_capflow_semantic_capacity_flow', width_mm=178.0,
        height_mm=58.0, min_font_pt=8.0, nodes=nodes, edges=edges,
        caption=SEMANTIC_CAPACITY_CAPTION)
```

Set `SEMANTIC_CAPACITY_CAPTION` to the approved caption in Section 6.3 of the
design spec.

- [ ] **Step 4: Run all paper-figure tests**

Expected: all tests PASS.

- [ ] **Step 5: Commit Figure 3 content**

```bash
rtk git add projects/OVCapFlow/tools/paper_figure_spec.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "feat: specify semantic-capacity mechanism figure"
```

### Task 5: Implement reproducible SVG, PDF, and PNG rendering

**Files:**
- Create: `projects/OVCapFlow/tools/generate_paper_figures.py`
- Modify: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Add export-format and live-text tests**

```python
from pathlib import Path

from projects.OVCapFlow.tools.generate_paper_figures import export_figure


def test_matplotlib_exports_vector_and_review_formats(tmp_path: Path):
    spec = FIGURE_SPECS['ov_capflow_t7_pipeline']
    paths = export_figure(spec, tmp_path, write_drawio=False)
    assert paths['svg'].read_text(encoding='utf-8').lstrip().startswith(
        '<?xml')
    assert '<text' in paths['svg'].read_text(encoding='utf-8')
    assert '<image' not in paths['svg'].read_text(encoding='utf-8')
    assert paths['pdf'].read_bytes().startswith(b'%PDF')
    assert paths['png'].read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_png_export_has_300_dpi_metadata(tmp_path: Path):
    from PIL import Image
    spec = FIGURE_SPECS['ov_capflow_t7_pipeline']
    paths = export_figure(spec, tmp_path, write_drawio=False)
    with Image.open(paths['png']) as image:
        dpi = image.info['dpi']
        assert abs(dpi[0] - 300.0) < 1.0
        assert abs(dpi[1] - 300.0) < 1.0
```

- [ ] **Step 2: Run and verify the renderer tests fail**

Expected: import failure for `generate_paper_figures`.

- [ ] **Step 3: Implement the Matplotlib canvas and typography configuration**

The renderer must:

```python
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from projects.OVCapFlow.tools.paper_figure_spec import (
    FIGURE_SPECS, PALETTE, Edge, FigureSpec, Node)


MM_PER_INCH = 25.4


def configure_matplotlib() -> None:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [
            'TeX Gyre Termes', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'axes.linewidth': 0.0,
    })


def render_spec(spec: FigureSpec):
    configure_matplotlib()
    spec.validate()
    fig = plt.figure(figsize=(
        spec.width_mm / MM_PER_INCH,
        spec.height_mm / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, spec.width_mm)
    ax.set_ylim(spec.height_mm, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    node_map = {node.id: node for node in spec.nodes}
    for node in sorted(spec.nodes, key=lambda item: item.kind != 'container'):
        draw_node(ax, node)
    for edge in spec.edges:
        draw_edge(ax, edge, node_map)
    return fig
```

- [ ] **Step 4: Implement containers, modules, text, and math nodes**

Implement `draw_node` dispatch for `container`, `module`, `text`, and `math`.
Containers use a transparent fill and place their label at the upper left;
modules use the role colour token; text nodes have no enclosing box; math
nodes pass labels through Matplotlib mathtext. Candidate modules set
`linestyle=(0, (3, 2))`.

```python
def draw_node(ax, node: Node) -> None:
    if node.kind == 'aerial':
        draw_aerial_glyph(ax, node)
        return
    if node.kind == 'rbox_output':
        draw_rbox_output_glyph(ax, node)
        return
    colour = PALETTE[node.role]
    if node.kind in {'text', 'math'}:
        ax.text(
            node.x + node.width / 2, node.y + node.height / 2,
            node.label, ha='center', va='center', fontsize=node.font_pt,
            color=colour.stroke,
            fontstyle='italic' if node.italic else 'normal')
        return
    is_container = node.kind == 'container'
    patch = FancyBboxPatch(
        (node.x, node.y), node.width, node.height,
        boxstyle='round,pad=0.45,rounding_size=1.5',
        facecolor='none' if is_container else colour.fill,
        edgecolor=colour.stroke, linewidth=0.85,
        linestyle=(0, (3, 2)) if node.dashed else 'solid',
        zorder=0 if is_container else 2)
    ax.add_patch(patch)
    ax.text(
        node.x + (1.5 if is_container else node.width / 2),
        node.y + (2.5 if is_container else node.height / 2),
        node.label, ha='left' if is_container else 'center',
        va='top' if is_container else 'center', fontsize=node.font_pt,
        color=PALETTE['ink'].stroke,
        fontstyle='italic' if node.italic else 'normal', zorder=3)
```

- [ ] **Step 5: Implement the vector aerial and rotated-output glyphs**

Implement `aerial` with a light-grey runway polygon, two building rectangles,
and three vehicle polygons. Implement `rbox_output` with a pale image-tile
rectangle and four unfilled rotated rectangles. Both glyphs use only
Matplotlib vector patches; do not call `imshow` or load an image file.

```python
def rotated_box_points(cx, cy, width, height, angle_deg):
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    points = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                 (width / 2, height / 2), (-width / 2, height / 2)):
        points.append((cx + x * cos_a - y * sin_a,
                       cy + x * sin_a + y * cos_a))
    return points


def draw_aerial_glyph(ax, node):
    ax.add_patch(FancyBboxPatch(
        (node.x, node.y), node.width, node.height,
        boxstyle='round,pad=0.2,rounding_size=1.0',
        facecolor='#F3F5F7', edgecolor=PALETTE['visual'].stroke,
        linewidth=0.85))
    ax.add_patch(Polygon(
        rotated_box_points(node.x + 7.5, node.y + 8.5, 3.0, 14.0, 58),
        closed=True, facecolor='#D4DAE0', edgecolor='none'))
    for dx, dy in ((2.0, 2.0), (10.0, 2.5)):
        ax.add_patch(FancyBboxPatch(
            (node.x + dx, node.y + dy), 3.0, 2.2,
            boxstyle='square,pad=0', facecolor='#AAB4BE', edgecolor='none'))
    for dx, dy in ((4.0, 12.0), (8.0, 10.0), (11.0, 13.0)):
        ax.add_patch(Polygon(
            rotated_box_points(node.x + dx, node.y + dy, 1.8, 0.8, -20),
            closed=True, facecolor=PALETTE['output'].stroke,
            edgecolor='none'))


def draw_rbox_output_glyph(ax, node):
    ax.add_patch(FancyBboxPatch(
        (node.x, node.y), node.width, node.height,
        boxstyle='round,pad=0.2,rounding_size=1.0',
        facecolor='#F7F8FA', edgecolor=PALETTE['output'].stroke,
        linewidth=0.85))
    for dx, dy, width, height, angle in (
            (4, 5, 5, 2, 15), (10, 6, 4, 1.8, -25),
            (6, 12, 4.5, 2.2, 40), (11, 14, 3.8, 1.7, 5)):
        ax.add_patch(Polygon(
            rotated_box_points(node.x + dx, node.y + dy,
                               width, height, angle),
            closed=True, fill=False, edgecolor=PALETTE['output'].stroke,
            linewidth=0.8))
```

- [ ] **Step 6: Implement arrows and the three-format exporter**

`draw_edge` connects the nearest horizontal or vertical box boundaries with
`FancyArrowPatch`; dotted edges use `linestyle=':'`. Implement
`export_figure` with:

```python
def connection_point(source: Node, target: Node):
    sx = source.x + source.width / 2
    sy = source.y + source.height / 2
    tx = target.x + target.width / 2
    ty = target.y + target.height / 2
    dx, dy = tx - sx, ty - sy
    if abs(dx) * source.height >= abs(dy) * source.width:
        return (source.x + source.width if dx >= 0 else source.x, sy)
    return (sx, source.y + source.height if dy >= 0 else source.y)


def draw_edge(ax, edge: Edge, nodes) -> None:
    start = connection_point(nodes[edge.source], nodes[edge.target])
    end = connection_point(nodes[edge.target], nodes[edge.source])
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=7,
        linewidth=0.9, color=PALETTE['training'].stroke,
        linestyle=':' if edge.style == 'dotted' else 'solid', zorder=1))
    if edge.label:
        ax.text(
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 1.0,
            edge.label, ha='center', va='bottom', fontsize=8.0,
            color=PALETTE['training'].stroke)
```

```python
def export_figure(spec, output_dir, write_drawio=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = render_spec(spec)
    paths = {
        suffix: output_dir / f'{spec.slug}.{suffix}'
        for suffix in ('svg', 'pdf', 'png')
    }
    fig.savefig(paths['svg'], format='svg', transparent=False)
    fig.savefig(paths['pdf'], format='pdf', transparent=False)
    fig.savefig(paths['png'], format='png', dpi=300, transparent=False)
    plt.close(fig)
    if write_drawio:
        paths['drawio'] = output_dir / f'{spec.slug}.drawio'
        write_drawio_file(spec, paths['drawio'])
    return paths
```

- [ ] **Step 7: Run renderer tests**

Expected: all renderer tests PASS.

- [ ] **Step 8: Commit the renderer**

```bash
rtk git add projects/OVCapFlow/tools/generate_paper_figures.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "feat: render OV-CapFlow paper figures"
```

### Task 6: Add editable diagrams.net output

**Files:**
- Modify: `projects/OVCapFlow/tools/generate_paper_figures.py`
- Modify: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Add draw.io XML tests**

```python
import xml.etree.ElementTree as ET


def test_drawio_export_is_editable_and_complete(tmp_path: Path):
    spec = FIGURE_SPECS['ov_capflow_full_architecture']
    paths = export_figure(spec, tmp_path, write_drawio=True)
    root = ET.parse(paths['drawio']).getroot()
    assert root.tag == 'mxfile'
    cells = root.findall('.//mxCell')
    vertex_ids = {cell.attrib['id'] for cell in cells
                  if cell.attrib.get('vertex') == '1'}
    assert {node.id for node in spec.nodes} <= vertex_ids
    candidate_cells = [cell for cell in cells
                       if cell.attrib.get('id') in {
                           'capacity', 'density', 'residual', 'null'}]
    assert candidate_cells
    assert all('dashed=1' in cell.attrib['style']
               for cell in candidate_cells)
```

- [ ] **Step 2: Run and verify the XML test fails**

Expected: FAIL because `write_drawio_file` is not implemented.

- [ ] **Step 3: Implement draw.io node styles**

Use `xml.etree.ElementTree`. The writer creates one uncompressed diagram with
root cells `0` and `1`; one `mxCell vertex=1` per node; one `mxCell edge=1` per
edge. Convert millimetres to draw.io pixels with `3.7795275591 px/mm`.

The style string must contain:

```python
def drawio_node_style(node: Node) -> str:
    colour = PALETTE[node.role]
    parts = [
        'rounded=0', 'whiteSpace=wrap', 'html=1',
        f'fillColor={colour.fill}', f'strokeColor={colour.stroke}',
        f'fontSize={node.font_pt}', 'fontFamily=Times New Roman',
        'align=center', 'verticalAlign=middle', 'strokeWidth=1',
    ]
    if node.dashed:
        parts.extend(['dashed=1', 'dashPattern=3 2'])
    if node.italic:
        parts.append('fontStyle=2')
    if node.kind == 'container':
        parts.extend(['fillOpacity=0', 'align=left', 'verticalAlign=top'])
    return ';'.join(parts) + ';'
```

This function is independently unit-testable and contains no XML mutation.

- [ ] **Step 4: Implement the uncompressed draw.io document and vertices**

Use `xml.etree.ElementTree`. Create one `mxfile`, one `diagram`, one
`mxGraphModel`, and root cells `0` and `1`. Add one `mxCell vertex=1` per node,
with an `mxGeometry` child. Convert millimetres to draw.io pixels with
`3.7795275591 px/mm`.

```python
import xml.etree.ElementTree as ET


PX_PER_MM = 3.7795275591


def build_drawio_tree(spec: FigureSpec):
    mxfile = ET.Element('mxfile', host='Electron', compressed='false')
    diagram = ET.SubElement(mxfile, 'diagram', name=spec.slug)
    model = ET.SubElement(
        diagram, 'mxGraphModel', dx='1200', dy='800', grid='1',
        gridSize='10', guides='1', tooltips='1', connect='1', arrows='1',
        fold='1', page='1', pageScale='1', pageWidth='1600',
        pageHeight='900', math='1', shadow='0')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')
    for node in sorted(spec.nodes, key=lambda item: item.kind != 'container'):
        cell = ET.SubElement(
            root, 'mxCell', id=node.id, value=node.label,
            style=drawio_node_style(node), vertex='1', parent='1')
        ET.SubElement(
            cell, 'mxGeometry',
            x=f'{node.x * PX_PER_MM:.3f}',
            y=f'{node.y * PX_PER_MM:.3f}',
            width=f'{node.width * PX_PER_MM:.3f}',
            height=f'{node.height * PX_PER_MM:.3f}', **{'as': 'geometry'})
    return mxfile, root
```

- [ ] **Step 5: Implement draw.io edges and file serialization**

Add one `mxCell edge=1` per edge. Edges use
`endArrow=block;endFill=1;rounded=0;`; dotted edges add
`dashed=1;dashPattern=1 3;`. Escape labels through ElementTree rather than
manual string concatenation. Serialize with an XML declaration and UTF-8.

```python
def write_drawio_file(spec: FigureSpec, path: Path) -> None:
    mxfile, root = build_drawio_tree(spec)
    for edge in spec.edges:
        style = 'endArrow=block;endFill=1;rounded=0;strokeWidth=1;'
        if edge.style == 'dotted':
            style += 'dashed=1;dashPattern=1 3;'
        cell = ET.SubElement(
            root, 'mxCell', id=edge.id, value=edge.label, style=style,
            edge='1', parent='1', source=edge.source, target=edge.target)
        ET.SubElement(cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
    ET.ElementTree(mxfile).write(
        path, encoding='utf-8', xml_declaration=True)
```

- [ ] **Step 6: Run all paper-figure tests**

Expected: all tests PASS.

- [ ] **Step 7: Commit draw.io support**

```bash
rtk git add projects/OVCapFlow/tools/generate_paper_figures.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "feat: export editable OV-CapFlow diagrams"
```

### Task 7: Add the generation CLI and output audit

**Files:**
- Modify: `projects/OVCapFlow/tools/generate_paper_figures.py`
- Modify: `tests/test_projects/ov_capflow/test_paper_figures.py`

- [ ] **Step 1: Add CLI and audit tests**

```python
from projects.OVCapFlow.tools.generate_paper_figures import (
    audit_outputs, generate_all)


def test_generate_all_writes_twelve_artifacts(tmp_path: Path):
    outputs = generate_all(tmp_path)
    assert len(outputs) == 12
    assert {path.suffix for path in outputs} == {
        '.drawio', '.svg', '.pdf', '.png'
    }
    assert audit_outputs(FIGURE_SPECS, outputs) == []
```

- [ ] **Step 2: Run and verify the integration test fails**

Expected: FAIL because `generate_all` and `audit_outputs` do not exist.

- [ ] **Step 3: Implement CLI generation and fail-closed auditing**

```python
def generate_all(output_dir):
    generated = []
    for spec in FIGURE_SPECS.values():
        generated.extend(export_figure(spec, output_dir).values())
    failures = audit_outputs(FIGURE_SPECS, generated)
    if failures:
        raise RuntimeError('\n'.join(failures))
    return tuple(generated)


def audit_outputs(specs, outputs):
    failures = []
    paths = {path.name: path for path in outputs}
    for spec in specs.values():
        for suffix in ('drawio', 'svg', 'pdf', 'png'):
            name = f'{spec.slug}.{suffix}'
            if name not in paths or paths[name].stat().st_size == 0:
                failures.append(f'missing or empty artifact: {name}')
        svg = paths[f'{spec.slug}.svg'].read_text(encoding='utf-8')
        if '<image' in svg:
            failures.append(f'raster asset embedded in {spec.slug}.svg')
        if '<text' not in svg:
            failures.append(f'text was converted to paths in {spec.slug}.svg')
    return failures
```

Add `argparse` with default output directory
`docs/figures/ov_capflow`. Run `generate_all` and print one line per artifact.

- [ ] **Step 4: Run the full focused tests**

Expected: all tests PASS.

- [ ] **Step 5: Commit the CLI and audit**

```bash
rtk git add projects/OVCapFlow/tools/generate_paper_figures.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "feat: audit paper figure exports"
```

### Task 8: Generate, inspect, and document the final artifacts

**Files:**
- Create: `docs/figures/ov_capflow/README.md`
- Create: `docs/figures/ov_capflow/ov_capflow_t7_pipeline.drawio`
- Create: `docs/figures/ov_capflow/ov_capflow_t7_pipeline.svg`
- Create: `docs/figures/ov_capflow/ov_capflow_t7_pipeline.pdf`
- Create: `docs/figures/ov_capflow/ov_capflow_t7_pipeline.png`
- Create: `docs/figures/ov_capflow/ov_capflow_full_architecture.drawio`
- Create: `docs/figures/ov_capflow/ov_capflow_full_architecture.svg`
- Create: `docs/figures/ov_capflow/ov_capflow_full_architecture.pdf`
- Create: `docs/figures/ov_capflow/ov_capflow_full_architecture.png`
- Create: `docs/figures/ov_capflow/ov_capflow_semantic_capacity_flow.drawio`
- Create: `docs/figures/ov_capflow/ov_capflow_semantic_capacity_flow.svg`
- Create: `docs/figures/ov_capflow/ov_capflow_semantic_capacity_flow.pdf`
- Create: `docs/figures/ov_capflow/ov_capflow_semantic_capacity_flow.png`

- [ ] **Step 1: Generate all artifacts**

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python \
  projects/OVCapFlow/tools/generate_paper_figures.py \
  --output-dir docs/figures/ov_capflow
```

Expected: twelve `generated:` lines and exit code 0.

- [ ] **Step 2: Run the focused test suite after generation**

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_paper_figures.py -q
```

Expected: all tests PASS.

- [ ] **Step 3: Inspect all three PNG previews at original resolution**

Open each PNG with the image viewer and check:

- no text overlaps or clipped labels;
- no arrow crosses a block or another arrow;
- Figure 1 has no amber candidate module;
- Figure 2 candidate modules are dashed amber and the substrate is visually
  quieter;
- Figure 3 equation is centred and remains readable;
- the figures remain understandable when viewed at approximately 50% scale.

If any check fails, change only the declarative coordinates/tokens, regenerate
all formats, and rerun the tests before continuing.

- [ ] **Step 4: Write the figure README with exact regeneration and captions**

The README must contain:

```markdown
# OV-CapFlow Paper Figures

These three figures use one declarative specification and are regenerated with:

`rtk /data/zcy/anaconda3/envs/mmdet/bin/python projects/OVCapFlow/tools/generate_paper_figures.py --output-dir docs/figures/ov_capflow`

## Evidence encoding

- solid outline: validated T7 substrate;
- dashed amber outline: implemented candidate excluded from current best T7;
- dotted grey arrow: training-only supervision.

## Publication use

Insert the PDF at `\textwidth` in a two-column paper. SVG is provided for
vector editing and web previews; PNG is review-only. All labels are at least
8 pt at the designed 178 mm width.

## Captions

**Validated T7 pipeline.** OV-CapFlow instantiates strict
vocabulary-conditioned oriented detection as a fixed-set transformer. Aerial
features and text tokens are encoded jointly, while 600 learned content
queries with learned 5-D rotated references are decoded directly into one
class decision and one rotated box per query. Grouped matching and denoising
queries are used only during training; inference retains one ordered Q600 set
without proposal ranking, global top-k selection, or NMS.

**Complete architecture.** The complete OV-CapFlow architecture regulates
semantic transport and scene capacity while retaining every query row. Each
decoder layer treats the unchanged parent output as transported evidence,
predicts continuous per-query capacity and global scene density, and applies
a parent-preserving native-query residual. The final null reservoir calibrates
background-heavy predictions; mass, density, and null losses are training-only.
Dashed modules are implemented candidates that are not enabled in the current
best T7 recipe.

**Semantic-capacity mechanism.** The zero-initialized semantic-capacity
residual preserves the parent decoder exactly before learning. The immutable
native query provides a persistent identity direction, the parent output
supplies transported visual-text evidence, and continuous capacity scales the
learned gated correction. A zero gate or zero capacity recovers the parent
output exactly, while inference keeps all queries rather than converting
capacity into a pruning decision.
```

- [ ] **Step 5: Commit the complete figure package**

```bash
rtk git add docs/figures/ov_capflow \
  projects/OVCapFlow/tools/paper_figure_spec.py \
  projects/OVCapFlow/tools/generate_paper_figures.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "docs: add OV-CapFlow methodology figures"
```

### Task 9: Run the final scientific and publication audit

**Files:**
- Verify only; modify the spec/generator only if an audit fails.

- [ ] **Step 1: Verify labels against active code and config**

```bash
rtk rg -n "num_queries=600|train_query_groups=3|matching_query_groups=3" \
  configs/ov_capflow/dotav2
rtk rg -n "FixedRotatedQueryInitializer|select_one_class_per_query|SemanticEvidenceFusion|ContinuousDensityCapacity|ExplicitNullReservoir" \
  projects/OVCapFlow/ov_capflow
```

Expected: each figure label maps to the active T7 config or to an implemented
candidate class named in the evidence legend.

- [ ] **Step 2: Verify no forbidden inference mechanism is drawn as active**

```bash
rtk rg -n "proposal top-k|global top-k|NMS|survivor" \
  docs/figures/ov_capflow/*.svg
```

Expected: these strings occur only in Figure 1's explicit `No ...` contract;
there is no block or arrow implementing them.

- [ ] **Step 3: Run focused and portable OV-CapFlow tests**

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow/test_paper_figures.py -q
rtk /data/zcy/anaconda3/envs/mmdet/bin/python -m pytest \
  -p no:cacheprovider tests/test_projects/ov_capflow \
  --ignore=tests/test_projects/ov_capflow/test_dotav2_real_batch.py \
  --ignore=tests/test_projects/ov_capflow/test_hrsc_real_batch.py -q
```

Expected: paper-figure tests pass; portable OV-CapFlow suite has no new
failures.

- [ ] **Step 4: Verify output inventory and worktree scope**

```bash
rtk /data/zcy/anaconda3/envs/mmdet/bin/python \
  projects/OVCapFlow/tools/generate_paper_figures.py \
  --output-dir docs/figures/ov_capflow
rtk git status --short
```

Expected: regeneration produces no diff; unrelated pre-existing untracked
files remain untouched.

- [ ] **Step 5: Record final verification commit if the audit required fixes**

If and only if Steps 1-4 required figure/spec changes:

```bash
rtk git add docs/figures/ov_capflow \
  projects/OVCapFlow/tools/paper_figure_spec.py \
  projects/OVCapFlow/tools/generate_paper_figures.py \
  tests/test_projects/ov_capflow/test_paper_figures.py
rtk git commit -m "fix: polish OV-CapFlow paper figures"
```

If no changes were required, do not create an empty commit.
