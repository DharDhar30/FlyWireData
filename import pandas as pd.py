import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from venn import venn
import seaborn as sns

# Setup plotting theme
sns.set_theme(style='white')
andy_theme = {
    'axes.grid': True, 'grid.linestyle': '--',
    'legend.framealpha': 1, 'legend.facecolor': 'white', 'legend.shadow': False,
    'legend.fontsize': 14, 'legend.title_fontsize': 14,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.labelsize': 12, 'axes.titlesize': 16, 'figure.dpi': 100
}
plt.rcParams.update(andy_theme)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

class FafbData:
    """Holds the four FlyWire/Codex tables (snapshot 783)."""
    def __init__(self, connections_csv, neuropil_csv, classification_csv, neurons_csv):
        self.connections = pd.read_csv(connections_csv)
        self.neuropils = (pd.read_csv(neuropil_csv)[['root_id', 'input synapses', 'output synapses']]
                          .rename(columns={'input synapses': 'input_synapses',
                                           'output synapses': 'output_synapses'}))
        cls = pd.read_csv(classification_csv)
        self.side = cls[['root_id', 'side']]
        self.classes = cls[['root_id', 'super_class', 'class']]
        self.neurons = pd.read_csv(neurons_csv)[['root_id', 'nt_type', 'group']]
        self.neurons_data = (self.neurons
                             .merge(self.side, on='root_id', how='outer')
                             .merge(self.neuropils, on='root_id', how='outer'))

class Neuralgroup:
    """A set of neurons plus their cached up/down-stream connectivity."""
    def __init__(self, subtype_df, fafbdata, name, min_syn=5):
        self.name = name
        self.min_syn_count = min_syn
        self.fafbdata = fafbdata

        df = subtype_df.loc[:, ~subtype_df.columns.str.contains('^Unnamed')].copy()
        self.neurons = df[['root_id']].drop_duplicates().reset_index(drop=True)
        self.region = self._detect_region(fafbdata.neurons)

        self.downconnections = self._edges(fafbdata.connections, 'down', min_syn)
        self.upconnections   = self._edges(fafbdata.connections, 'up',   min_syn)

        self.postneurons = self._partners(self.downconnections, 'post_root_id')
        self.preneurons  = self._partners(self.upconnections,  'pre_root_id')

        self_ids = set(self.neurons['root_id'])
        self.downflow = self.postneurons[~self.postneurons['root_id'].isin(self_ids)].reset_index(drop=True)
        self.upflow   = self.preneurons[~self.preneurons['root_id'].isin(self_ids)].reset_index(drop=True)

    def _detect_region(self, neurons_df):
        groups = neurons_df.loc[neurons_df['root_id'].isin(self.neurons['root_id']), 'group'].dropna()
        regions = set()
        for g in groups:
            regions.update(str(g).split('.'))
        return regions

    def _projection(self, neuropil_side):
        if pd.isna(neuropil_side):
            return 'nonlocal'
        npil = neuropil_side.replace('_R', '').replace('_L', '')
        return 'local' if npil in self.region else 'nonlocal'

    def _edges(self, connections_df, direction, min_syn):
        key = 'pre_root_id' if direction == 'down' else 'post_root_id'
        conn = connections_df[['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type']]
        edges = (self.neurons
                 .merge(conn, left_on='root_id', right_on=key, how='inner')
                 .drop(columns='root_id')
                 .query('syn_count >= @min_syn')
                 .copy())
        edges['locality'] = edges['neuropil'].apply(self._projection)
        return edges.reset_index(drop=True)

    @staticmethod
    def _partners(edges, col):
        if edges.empty:
            return pd.DataFrame({'root_id': pd.Series(dtype='int64')})
        return (edges.groupby(col)
                .agg({'syn_count': 'sum'})
                .reset_index()[[col]]
                .rename(columns={col: 'root_id'}))

    def DownstreamNeurons(self, connections_df=None, min_syn=None):
        if min_syn is None:
            min_syn = self.min_syn_count
        if connections_df is None and min_syn >= self.min_syn_count:
            return self.downconnections.query('syn_count >= @min_syn').reset_index(drop=True)
        src = connections_df if connections_df is not None else self.fafbdata.connections
        return self._edges(src, 'down', min_syn)

    def UpstreamNeurons(self, connections_df=None, min_syn=None):
        if min_syn is None:
            min_syn = self.min_syn_count
        if connections_df is None and min_syn >= self.min_syn_count:
            return self.upconnections.query('syn_count >= @min_syn').reset_index(drop=True)
        src = connections_df if connections_df is not None else self.fafbdata.connections
        return self._edges(src, 'up', min_syn)

class Connectsets:
    """Multi-order connectivity between a list of Neuralgroups."""
    def __init__(self, neuralgroup_list, fafbdata,
                 downstream_order=1, upstream_order=0,
                 order1_min_syn=5, order2_min_syn=10):
        self.maingroups = {ng.name: ng for ng in neuralgroup_list}
        self.fafbdata = fafbdata
        self.downstream_order = downstream_order
        self.upstream_order = upstream_order
        self.order1_min_syn = order1_min_syn
        self.order2_min_syn = order2_min_syn
        self._all_main_ids = set(pd.concat([ng.neurons for ng in neuralgroup_list])['root_id'])

        self.out1_connections, self.out1_groups = {}, {}
        self.out2_connections, self.out2_groups = {}, {}
        self.in1_connections,  self.in1_groups  = {}, {}
        self.in2_connections,  self.in2_groups  = {}, {}

        if downstream_order >= 1:
            self.out1_connections, self.out1_groups = self._first_order('down', order1_min_syn)
        if downstream_order >= 2:
            self.out2_connections, self.out2_groups = self._second_order('down', order2_min_syn)
        if upstream_order >= 1:
            self.in1_connections, self.in1_groups = self._first_order('up', order1_min_syn)
        if upstream_order >= 2:
            self.in2_connections, self.in2_groups = self._second_order('up', order2_min_syn)

    def _first_order(self, stream, min_syn):
        connectivity, groups = {}, {}
        for name, ng in self.maingroups.items():
            if stream == 'down':
                edges, partner_col = ng.DownstreamNeurons(min_syn=min_syn), 'post_root_id'
            else:
                edges, partner_col = ng.UpstreamNeurons(min_syn=min_syn), 'pre_root_id'
            edges = edges[~edges[partner_col].isin(self._all_main_ids)].reset_index(drop=True)
            partners = Neuralgroup._partners(edges, partner_col)
            groups[name] = Neuralgroup(partners, self.fafbdata, name, min_syn=min_syn)
            connectivity[name] = edges
        return connectivity, groups

    def _second_order(self, stream, min_syn):
        order1_groups = self.out1_groups if stream == 'down' else self.in1_groups
        if not order1_groups:
            raise ValueError('First-order groups not built.')
        all_order1 = set(pd.concat([ng.neurons for ng in order1_groups.values()])['root_id'])
        connectivity, groups = {}, {}
        for name, ng in order1_groups.items():
            if stream == 'down':
                edges, partner_col = ng.DownstreamNeurons(min_syn=min_syn), 'post_root_id'
            else:
                edges, partner_col = ng.UpstreamNeurons(min_syn=min_syn), 'pre_root_id'
            edges = edges[~edges[partner_col].isin(self._all_main_ids)]
            edges = edges[~edges[partner_col].isin(all_order1)]
            edges = edges.drop_duplicates().reset_index(drop=True)
            partners = Neuralgroup._partners(edges, partner_col)
            groups[name] = Neuralgroup(partners, self.fafbdata, name, min_syn=min_syn)
            connectivity[name] = edges
        return connectivity, groups

    def _groups_at(self, lvl):
        mapping = {0: self.maingroups, 1: self.out1_groups, 2: self.out2_groups,
                   -1: self.in1_groups, -2: self.in2_groups}
        return mapping.get(lvl)

# --- Initialize Dataset & DSK Group ---
fafb = FafbData(connections_csv='connections_783.csv',
                neuropil_csv='neuropil_synapse_table_783.csv',
                classification_csv='classification_783.csv',
                neurons_csv='neurons_783.csv')

# Load the DSK IDs you exported from FlyWire
DSK = Neuralgroup(pd.read_csv('DSK_r_id.csv'), fafb, 'DSK', min_syn=5)

print(f'DSK group has {len(DSK.neurons)} neurons')
print(f'DSK makes {len(DSK.downconnections)} output edges (>=5 syn)')

# --- Edges and Partner Extractions ---
all_downstream_edges = DSK.DownstreamNeurons(min_syn=5)
all_upstream_edges = DSK.UpstreamNeurons(min_syn=5)

print(f"Total downstream connection edges: {len(all_downstream_edges)}")
print(f"Total upstream connection edges: {len(all_upstream_edges)}")

# --- Upstream Neurotransmitter Profile ---
upstream_nt_summary = (
    all_upstream_edges.groupby("nt_type")
    .agg(
        edge_count=("post_root_id", "count"),
        total_synapses=("syn_count", "sum"),
        unique_pre_neurons=("pre_root_id", "nunique"),
    )
    .reset_index()
    .sort_values(by="total_synapses", ascending=False)
)
upstream_nt_summary["synapse_pct"] = (
    upstream_nt_summary["total_synapses"] / upstream_nt_summary["total_synapses"].sum()
) * 100

print("\n--- 1. UPSTREAM NEUROTRANSMITTER PROFILE ---")
display(upstream_nt_summary)

plt.figure(figsize=(8, 4))
sns.barplot(data=upstream_nt_summary, x="nt_type", y="total_synapses", palette="viridis")
plt.title("Upstream Input by Neurotransmitter Type (DSK)")
plt.xlabel("Neurotransmitter Type")
plt.ylabel("Total Synapses Received")
plt.show()

# --- Neuropil Breakdown ---
neuropil_breakdown = (
    all_downstream_edges.groupby(["neuropil", "locality"])
    .agg(
        edge_count=("pre_root_id", "count"),
        total_synapses=("syn_count", "sum"),
        unique_post_neurons=("post_root_id", "nunique"),
    )
    .reset_index()
    .sort_values(by="total_synapses", ascending=False)
)
print("\n--- 2. NEUROPIL FLOW BREAKDOWN ---")
display(neuropil_breakdown.head(10))

# --- Excel Export ---
all_downstream_edges['pre_root_id'] = all_downstream_edges['pre_root_id'].astype(str)
all_downstream_edges['post_root_id'] = all_downstream_edges['post_root_id'].astype(str)
all_upstream_edges['pre_root_id'] = all_upstream_edges['pre_root_id'].astype(str)
all_upstream_edges['post_root_id'] = all_upstream_edges['post_root_id'].astype(str)

output_file = 'DSK_connections.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    all_downstream_edges.to_excel(writer, sheet_name='Downstream', index=False)
    all_upstream_edges.to_excel(writer, sheet_name='Upstream', index=False)

print(f"\nSaved connectivity data to {output_file} successfully!")