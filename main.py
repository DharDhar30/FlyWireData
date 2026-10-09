import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Setup plotting style
sns.set_theme(style='white')
andy_theme = {'axes.grid': True, 'grid.linestyle': '--',
              'legend.framealpha': 1, 'legend.facecolor': 'white', 'legend.shadow': False,
              'legend.fontsize': 14, 'legend.title_fontsize': 14,
              'xtick.labelsize': 8, 'ytick.labelsize': 8,
              'axes.labelsize': 12, 'axes.titlesize': 16, 'figure.dpi': 100}
plt.rcParams.update(andy_theme)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

class FafbData:
    """Holds FlyWire connectivity and metadata tables, adapted for your connection export columns."""
    def __init__(self, connections_csv, search_results_csv):
        raw_conn = pd.read_csv(connections_csv)
        # Maps your connection columns ('From', 'To', 'Neuropil', 'Synapses', 'Neuro Transmitter')
        self.connections = raw_conn.rename(columns={
            'From': 'pre_root_id',
            'To': 'post_root_id',
            'Neuropil': 'neuropil',
            'Synapses': 'syn_count',
            'Neuro Transmitter': 'nt_type'
        })
        
        cls = pd.read_csv(search_results_csv)
        self.side = cls[['root_id', 'side']] if 'side' in cls.columns else pd.DataFrame({'root_id': cls['root_id'], 'side': 'unknown'})
        self.classes = cls[['root_id', 'super_class', 'class']] if {'super_class', 'class'}.issubset(cls.columns) else pd.DataFrame({'root_id': cls['root_id'], 'super_class': 'descending', 'class': 'DSK'})
        self.neurons = cls[['root_id', 'nt_type', 'cell_type']].rename(columns={'cell_type': 'group'}) if {'nt_type', 'cell_type'}.issubset(cls.columns) else pd.DataFrame({'root_id': cls['root_id'], 'nt_type': 'SER', 'group': 'DSK'})

class Neuralgroup:
    """A set of neurons plus their cached up/down-stream connectivity."""
    def __init__(self, subtype_df, fafbdata, name, min_syn=2):
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
        if 'group' not in neurons_df.columns:
            return set()
        groups = neurons_df.loc[neurons_df['root_id'].isin(self.neurons['root_id']), 'group'].dropna()
        regions = set()
        for g in groups:
            regions.update(str(g).split('.'))
        return regions

    def _projection(self, neuropil_side):
        if pd.isna(neuropil_side):
            return 'nonlocal'
        npil = str(neuropil_side).replace('_R', '').replace('_L', '')
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

    def ReciprocalPartners(self, min_syn=None):
        down = self.DownstreamNeurons(min_syn=min_syn)
        up = self.UpstreamNeurons(min_syn=min_syn)
        self_ids = set(self.neurons['root_id'])

        down_ids = set(down['post_root_id']) - self_ids
        up_ids = set(up['pre_root_id']) - self_ids
        recip_ids = down_ids & up_ids

        down_agg = (down[down['post_root_id'].isin(recip_ids)]
                    .groupby('post_root_id')['syn_count'].sum().rename('down_syn'))
        up_agg = (up[up['pre_root_id'].isin(recip_ids)]
                  .groupby('pre_root_id')['syn_count'].sum().rename('up_syn'))

        out = pd.concat([down_agg, up_agg], axis=1).fillna(0).astype(int)
        out['total_syn'] = out['down_syn'] + out['up_syn']
        return out.sort_values('total_syn', ascending=False).reset_index().rename(columns={'index': 'root_id'})

    def TopSynapsePartners(self, min_syn=None):
        down = self.DownstreamNeurons(min_syn=min_syn)
        up = self.UpstreamNeurons(min_syn=min_syn)
        self_ids = set(self.neurons['root_id'])

        down_agg = (down[~down['post_root_id'].isin(self_ids)]
                    .groupby('post_root_id')['syn_count'].sum().rename('down_syn'))
        up_agg = (up[~up['pre_root_id'].isin(self_ids)]
                  .groupby('pre_root_id')['syn_count'].sum().rename('up_syn'))

        out = pd.concat([down_agg, up_agg], axis=1).fillna(0).astype(int)
        out['total_syn'] = out['down_syn'] + out['up_syn']
        return out.sort_values('total_syn', ascending=False).reset_index().rename(columns={'index': 'root_id'})

class Connectsets:
    """Multi-order connectivity between Neuralgroups."""
    def __init__(self, neuralgroup_list, fafbdata,
                 downstream_order=1, upstream_order=0,
                 order1_min_syn=2, order2_min_syn=2):
        self.maingroups = {ng.name: ng for ng in neuralgroup_list}
        self.fafbdata = fafbdata
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

# --- Initialize Dataset & DSK Group ---
fafb = FafbData(connections_csv='connections.csv', search_results_csv='search_results_DSK.csv')
DSK = Neuralgroup(pd.read_csv('search_results_DSK.csv'), fafb, 'DSK', min_syn=2)

print(f'DSK has {len(DSK.neurons)} neurons')
print(f'DSK makes {len(DSK.downconnections)} output edges')

all_downstream_edges = DSK.DownstreamNeurons(min_syn=2)
all_upstream_edges = DSK.UpstreamNeurons(min_syn=2)

# --- Upstream NT Summary ---
upstream_nt_summary = (
    all_upstream_edges.groupby("nt_type")
    .agg(
        edge_count=("pre_root_id", "count"),
        total_synapses=("syn_count", "sum"),
        unique_pre_neurons=("pre_root_id", "nunique"),
    )
    .reset_index()
    .sort_values(by="total_synapses", ascending=False)
)
upstream_nt_summary["synapse_pct"] = (
    upstream_nt_summary["total_synapses"] / upstream_nt_summary["total_synapses"].sum()
) * 100

print("\n--- Upstream NT Summary ---")
print(upstream_nt_summary.to_string(index=False))

# --- Neuropil Breakdown ---
neuropil_breakdown = (
    all_downstream_edges.groupby(["neuropil", "locality"])
    .agg(
        edge_count=("post_root_id", "count"),
        total_synapses=("syn_count", "sum"),
        unique_post_neurons=("post_root_id", "nunique"),
    )
    .reset_index()
    .sort_values(by="total_synapses", ascending=False)
)

print("\n--- Neuropil Breakdown ---")
print(neuropil_breakdown.head(10).to_string(index=False))

# --- Multi-order Network Expansion ---
cs_2nd = Connectsets([DSK], fafb, downstream_order=2, upstream_order=2, order1_min_syn=2, order2_min_syn=2)
print("\nMulti-order Connectsets initialized successfully!")

# --- Excel Export ---
all_downstream_edges['pre_root_id'] = all_downstream_edges['pre_root_id'].astype(str)
all_downstream_edges['post_root_id'] = all_downstream_edges['post_root_id'].astype(str)
all_upstream_edges['pre_root_id'] = all_upstream_edges['pre_root_id'].astype(str)
all_upstream_edges['post_root_id'] = all_upstream_edges['post_root_id'].astype(str)

output_file = 'DSK_full_analysis.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    all_downstream_edges.to_excel(writer, sheet_name='Downstream', index=False)
    all_upstream_edges.to_excel(writer, sheet_name='Upstream', index=False)
    upstream_nt_summary.to_excel(writer, sheet_name='NT_Summary', index=False)
    neuropil_breakdown.to_excel(writer, sheet_name='Neuropil_Summary', index=False)
    if 'DSK' in cs_2nd.out1_connections:
        cs_2nd.out1_connections['DSK'].astype(str).to_excel(writer, sheet_name='1st_Order_Downstream', index=False)
    if 'DSK' in cs_2nd.out2_connections:
        cs_2nd.out2_connections['DSK'].astype(str).to_excel(writer, sheet_name='2nd_Order_Downstream', index=False)

print(f"\nSuccessfully generated and saved export: {output_file}!")