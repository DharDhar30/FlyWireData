import pandas as pd
import numpy as np

# Load and rename columns consistently
raw = pd.read_csv('connections.csv', dtype=str)
c = raw.rename(columns={
    'From': 'pre_root_id',
    'To': 'post_root_id',
    'Neuropil': 'neuropil',
    'Synapses': 'syn_count',
    'Neuro Transmitter': 'nt_type',
})

cls = pd.read_csv('search_results_DSK.csv', dtype=str)
main_ids = set(cls['root_id'])

# --- Direct downstream edges (DSK -> ANY target, including other DSK cells) ---
down = (c[c['pre_root_id'].isin(main_ids)][
    ['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type']]
    .groupby(['post_root_id', 'neuropil', 'nt_type'], as_index=False)['syn_count'].sum())
print('Direct downstream edges:', len(down))

order1_targets = set(down['post_root_id'])
print('order1 target count:', len(order1_targets))


def downstream_edges(src_ids):
    mask = c['pre_root_id'].isin(src_ids)
    e = c[mask][['pre_root_id', 'post_root_id', 'neuropil', 'syn_count', 'nt_type']].copy()
    e = e.groupby(['post_root_id', 'neuropil', 'nt_type'], as_index=False)['syn_count'].sum()
    return e


# --- 2nd order downstream ---
o2 = downstream_edges(order1_targets)
print('o2 edges:', len(o2))
print('o2 rows:')
print(o2.to_string(index=False))

# --- Layer aggregation by neuropil ---
def layer_neuropil_agg(edges, label):
    if edges.empty:
        return pd.DataFrame(columns=['neuropil', 'syn_count'])
    g = edges.groupby('neuropil')['syn_count'].sum().reset_index()
    g['layer'] = label
    return g


df = pd.concat(
    [layer_neuropil_agg(down, '1st_Order_Downstream'),
     layer_neuropil_agg(o2, '2nd_Order_Downstream')],
    ignore_index=True)
df['syn_count'] = pd.to_numeric(df['syn_count'], errors='coerce').fillna(0).astype(int)

print()
print(df.to_string(index=False))

# --- Validation ---
print()
print('=== VALIDATION ===')
for layer in ['1st_Order_Downstream', '2nd_Order_Downstream']:
    sub = df[df['layer'] == layer]
    total = sub['syn_count'].sum()
    pct = (sub['syn_count'] / total * 100).round(4).fillna(0)
    print(f"{layer}: total_synapses={total}, rows={len(sub)}, pct_sum={pct.sum()}")
# --- Validation (appended at end) ---
print()
print('=== VALIDATION ===')
for layer in ['1st_Order_Downstream', '2nd_Order_Downstream']:
    sub = df[df['layer'] == layer].copy()
    sub['syn_count'] = pd.to_numeric(sub['syn_count'], errors='coerce').fillna(0).astype(int)
    total = int(sub['syn_count'].sum())
    pct = (sub['syn_count'] / total * 100).round(4).fillna(0)
    print(f"{layer}: total_synapses={total}, rows={len(sub)}, pct_sum={pct.sum()}")
    print(f"  Top neuropils: {sub.nlargest(5, 'syn_count')['neuropil'].tolist()}")
    print(f"  Neuropil breakdown (pct): {pct.round(1).to_dict()}")

    print(f"  Top neuropils: {sub.nlargest(5, 'syn_count')['neuropil'].tolist()}")
