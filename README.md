# FlyWireData - Drosophila Connectomics Analysis

A Python-based toolkit for analyzing neural connectivity data from the FlyWire project (Drosophila melanogaster, adult brain). This repository ingests pre-existing CSV exports and produces a structured network analysis of the DSK (sulfakinin-producing) neuron population, including multi-order connectivity expansion and Excel export.

---

## Contents

1. [Data Files](#-data-files)
2. [Class Hierarchy & Data Model](#-class-hierarchy--data-model)
3. [Field Definitions](#-field-definitions)
4. [Schema Relationships](#-schema-relationships)
5. [Pipeline Overview](#-pipeline-overview)
6. [Output Files](#-output-files)
7. [Usage](#-usage)

---

## Data Files

| File | Type | Description |
|---|---|---|
| connections.csv | CSV | Raw synaptic connectivity edges between neurons. Two columns: From and To hold FlyWire root IDs (neuron identifiers). |
| search_results_DSK.csv | CSV | Annotated metadata for each neuron in the dataset. Contains 14 columns describing cell identity, classification, and connectivity. |

> **Note:** All root IDs are FlyWire neuron identifiers (integer FlyWire IDs). These are stable, unique identifiers for individual neurons in the FlyWire volume.

---

## Class Hierarchy & Data Model

The codebase is organized around three core classes:

    FafbData          -> Holds raw connectivity & metadata tables
        └── Neuralgroup -> Represents a set of neurons + cached up/down-stream edges
                └── Connectsets -> Multi-order connectivity between Neuralgroups

### FafbData (Lines 18-34)

Encapsulates the raw data layer. In __init__(), it reads both CSVs and renames columns to canonical internal names:

| Input CSV column | Internal attribute | Renamed to |
|---|---|---|
| From | connections | pre_root_id |
| To | connections | post_root_id |
| Neuropil | connections | neuropil |
| Synapses | connections | syn_count |
| Neuro Transmitter | connections | nt_type |
| root_id | side | side |
| root_id | classes | super_class, class |
| root_id | neurons | nt_type, cell_type (renamed group) |

> **Internal column naming rationale:** pre_root_id and post_root_id are used because in the FlyWire hemibrain nomenclature, the presynaptic side and postsynaptic side are tracked separately from the "From/To" raw export. pre_ = source neuron, post_ = target neuron.

### Neuralgroup (Lines 36-138)

Represents a set of neurons and all their cached connectivity. Key attributes:

| Attribute | Type | Description |
|---|---|---|
| name | str | Group name (e.g., "DSK") |
| min_syn_count | int | Minimum synapse threshold for edge inclusion (default 2) |
| fafbdata | FafbData | Reference to the parent data holder |
| neurons | pd.DataFrame | List of root IDs in this group (root_id column) |
| region | set | Unique brain regions derived from the group column (e.g., {"SLP", "GNG", "AVLP"}) |
| downconnections | pd.DataFrame | All outgoing synapses (presynaptic partners) from this group |
| upconnections | pd.DataFrame | All incoming synapses (postsynaptic partners) to this group |
| postneurons | pd.DataFrame | Downstream partners (postsynaptic targets, summed by syn_count) |
| preneurons | pd.DataFrame | Upstream partners (presynaptic sources, summed by syn_count) |
| downflow | pd.DataFrame | Downstream partners outside the current group |
| upflow | pd.DataFrame | Upstream partners outside the current group |

**Key methods:**

- DownstreamNeurons(connections_df=None, min_syn=None) -> Returns all outgoing edges (soma to postsynaptic targets).
- UpstreamNeurons(connections_df=None, min_syn=None) -> Returns all incoming edges (presynaptic sources to soma).
- ReciprocalPartners(min_syn=None) -> Returns neurons that both receive from and project to this group (bidirectional partners), with per-direction synapse counts.
- TopSynapsePartners(min_syn=None) -> Returns neurons ranked by total synapse count (incoming + outgoing).

**Edge computation detail (_edges method):**

    self.neurons (as 'root_id' column)
        X connections_df
            on 'root_id' = 'pre_root_id' (for 'down' direction)
## Field Definitions

### connections.csv (Raw)

| Column | Type | Description |
|---|---|---|
| From | int | Pre-synaptic neuron root ID (source of the connection) |
| To | int | Post-synaptic neuron root ID (target of the connection) |
| Neuropil | str | Brain region name (e.g., SLP_R, MB_VL_L, LH_R). Suffix _R or _L = right/left hemisphere. |
| Synapses | int | Number of observed synaptic contacts for this edge |
| Neuro Transmitter | str | Neurotransmitter name (e.g., DA = dopamine, SER = serotonin, ACh = acetylcholine) |

### search_results_DSK.csv (Annotated)

| Column | Type | Description |
|---|---|---|
| root_id | int | Unique FlyWire neuron identifier (primary key) |
| label | str | Cell-type label; may be a list-like string (e.g., ['sulfakinin producing neuron MP1b'; 'DSK-MP1B neuron'; 'DNp']) |
| name | str | Short name (e.g., SLP.GNG.3, SLP.SMP.118) |
| nt_type | str | Neurotransmitter type (e.g., SER, DA) |
| flow | str | Neural flow direction: efferent (output to downstream brain regions) or intrinsic (local circuit) |
| super_class | str | Broad category: default descending for this dataset |
| class | str | Sub-class classification: default DSK for this dataset |
| sub_class | str | Empty by default (not populated in current export) |
| cell_type | str | Canonical cell-type identifier (e.g., ['DNp62'; 'DSKMP1B']) |
| hemilineage | str | Hemilineage tag (empty by default) |
| nerve | str | Nerve tag (e.g., CV) |
| connectivity_tag | str | List-like connectivity motif tag (e.g., ['rich_club'; 'reciprocal'; 'feedforward_loop_participant'; '3_cycle_participant']) |
| side | str | Brain hemisphere: left or right |
| input_synapses | int | Total synaptic input count onto this neuron from all sources (pre-computed) |
| output_synapses | int | Total synaptic output count from this neuron to all targets (pre-computed) |

        -> filter syn_count >= min_syn
        -> add 'locality' column: 'local' if neuropil is in this group's region, else 'nonlocal'

### Connectsets (Lines 140-193)

Represents multi-order connectivity between Neuralgroup objects. It computes up to 2nd-order (and configurable-order) connections in both down and up streams.

| Attribute | Type | Description |
|---|---|---|
| maingroups | dict | {group_name: Neuralgroup} lookup |
| fafbdata | FafbData | Reference to raw data |
| _all_main_ids | set | All root IDs across all groups |
| out1_connections | dict | {group_name: DataFrame} 1st-order downstream edges |
| out1_groups | dict | {group_name: Neuralgroup} 1st-order downstream partners |
| out2_connections | dict | {group_name: DataFrame} 2nd-order downstream edges |
| out2_groups | dict | {group_name: Neuralgroup} 2nd-order downstream partners |
| in1_connections | dict | {group_name: DataFrame} 1st-order upstream edges |
| in1_groups | dict | {group_name: Neuralgroup} 1st-order upstream partners |
| in2_connections | dict | {group_name: DataFrame} 2nd-order upstream edges |
| in2_groups | dict | {group_name: Neuralgroup} 2nd-order upstream partners |

**Method logic:**
## Schema Relationships

    connections.csv
      ├── From  ->  pre_root_id  (projected into Neuralgroup.neurons)
      │
      ├── To    ->  post_root_id (projected into Neuralgroup.neurons)

    search_results_DSK.csv
      ├── root_id  ->  unique identifier (shared with connections.csv -> pre/post columns)
      ├── nt_type  ->  neurotransmitter (cached into Neuralgroup.neurons as 'group')
      ├── cell_type -> renamed to 'group' in Neuralgroup constructor
      └── side     ->  used to determine hemibrain laterality (not used in connection logic directly)

The FafbData class joins these two sources:

- Columns from connections.csv are renamed and stored in self.connections.
- Columns from search_results_DSK.csv are selectively extracted into self.side, self.classes, and self.neurons.

---

## Pipeline Overview

    1. Read connections.csv -> FafbData.connections
    2. Read search_results_DSK.csv -> FafbData.neurons, .classes, .side
    3. Create Neuralgroup('DSK') with min_syn=2
    4. Compute all downstream edges (output to postsynaptic partners)
    5. Compute all upstream edges (input from presynaptic partners)
    6. Summarize upstream neurotransmitter distribution
    7. Summarize downstream neuropil breakdown
    8. Initialize Connectsets with 2nd-order expansion (both directions)
    9. Export all results to DSK_full_analysis.xlsx

### Key Filtering

- Minimum synapse threshold: min_syn=2 (edges with <2 synapses are excluded).
- Local vs. nonlocal: locality column marks whether a neuropil connection is within the group's known regions (local) or outside them (nonlocal).
- Non-newline partners: downflow and upflow exclude partners that are already inside the group (self_ids).

---

## Output Files

The pipeline generates DSK_full_analysis.xlsx with these sheets:

| Sheet | Content |
|---|---|
| Downstream | All outgoing edges from the DSK group (post_root_id, syn_count, neuropil, locality, nt_type) |
| Upstream | All incoming edges to the DSK group (pre_root_id, syn_count, neuropil, locality, nt_type) |
| NT_Summary | Summary of upstream neurotransmitters: edge count, total synapses, unique pre-neurons, synapse percentage |
| Neuropil_Summary | Breakdown of downstream connections by neuropil and locality |
| 1st_Order_Downstream | 1st-order downstream partners (2nd-order expansion, computed but only exported if present) |
| 2nd_Order_Downstream | 2nd-order downstream partners |

---

## Usage

    pip install pandas numpy matplotlib seaborn openpyxl
    python main.py

The script will:

1. Load the two CSV files.
2. Print summary statistics to the console.
3. Generate DSK_full_analysis.xlsx.

### Variables You Can Tune

- min_syn (in Neuralgroup and method calls) -> minimum synapse threshold (default 2).
- downstream_order / upstream_order (in Connectsets) -> number of orders to compute (0, 1, or 2).
- order1_min_syn / order2_min_syn -> synapse thresholds for 1st/2nd order respectively.

## Quick Reference: Key Data Model Concepts

| Concept | Meaning |
|---|---|
| root_id | FlyWire neuron identifier (primary key) |
| pre_root_id | Presynaptic neuron (source) |
| post_root_id | Postsynaptic neuron (target) |
| syn_count | Number of synaptic connections on an edge |
| neuropil | Brain region (e.g., SLP_R, MB_VL_L) |
| nt_type | Neurotransmitter type (DA, SER, etc.) |
| locality | local if connection is within the group's regions, nonlocal otherwise |
| downconnections | Outgoing edges (this group to postsynaptic partners) |
| upconnections | Incoming edges (presynaptic partners to this group) |
| postneurons | Post-synaptic partners (cached sum of synapses) |
| preneurons | Pre-synaptic partners (cached sum of synapses) |
| flow | efferent (output) or intrinsic (local) |
| super_class / class | Broad classification of neuron type |

---

*Built for FlyWire hemibrain connectome analysis | Drosophila melanogaster | v1.2*

---

## Biological Context

This project focuses on DSK neurons - sulfakinin-producing neurons in the Drosophila central brain. These neurons:

- Are efferent (flow=efferent) - they relay signals from the central brain to neuropil regions.
- Are classified as descending (super_class=descending).
- Express the neurotransmitter serotonin (SER) or dopamine (DA).
- Are part of the FlyWire hemibrain connectome (v1.2).

The name column encodes brain regions (e.g., SLP.GNG.3, SLP.SMP.118), where:
- SLP = Superior Lateral Protocerebrum
- GNG = Goler's Neuropil / Gnathal regions
- SMP = Superior Medial Protocerebrum
- AVLP = Anterior Ventrolateral Protocerebrum

connectivity_tag values like rich_club, reciprocal, feedforward_loop_participant, and 3_cycle_participant hint at network motifs these neurons participate in.


