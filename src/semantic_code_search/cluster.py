import gzip
import os
import pickle
from semantic_code_search.embed import do_embed
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from textwrap import indent


from collections import defaultdict
import numpy as np
from statistics import mean
from sklearn.cluster import AgglomerativeClustering

def _get_clusters(dataset, distance_threshold):
    embeddings = dataset.get('embeddings')
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    dataset['embeddings'] = embeddings

    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=distance_threshold,
                                               compute_distances=True)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_distances = clustering_model.distances_
    cluster_children = clustering_model.children_

    clustered_functions = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_assignment):
        ds_entry = dataset.get('functions')[idx]
        ds_entry['idx'] = idx
        clustered_functions[cluster_id].append(ds_entry)

    clusters = []
    for cluster_id, functions in clustered_functions.items():
        if len(functions) > 1:
            fx_idx = functions[0].get('idx')
            distances = [cluster_distances[i] for i, cc in enumerate(cluster_children)
                         if cc.tolist() == [fx_idx, f.get('idx')]
                         for f in functions[1:]]
            avg_distance = mean(distances) if distances else 0
            clusters.append({'avg_distance': avg_distance, 'functions': functions})

    return clusters

def do_cluster(args,model):
    embeddings_file = os.path.join(args.path_to_repo, '.embeddings')
    if not os.path.isfile(embeddings_file):
        print(f'Embeddings not found in {args.path_to_repo}. Generating embeddings now...')
        do_embed(args, model)
    with open(embeddings_file, 'rb') as f:
        dataset = pickle.load(f)
        if dataset.get('model_name') != args.model_name_or_path:
            print('Model name mismatch. Regenerating embeddings.')
            dataset = do_embed(args, model)
        clusters = _get_clusters(dataset, args.cluster_max_distance)
        filtered_clusters = [c for c in clusters
                             if not (args.cluster_ignore_identincal and c.get('avg_distance') == 0)
                             and not any(len(f.get('text').split('\n')) <= args.cluster_min_lines for f in c.get('functions'))
                             and len(c.get('functions')) >= args.cluster_min_cluster_size]
        for i, c in enumerate(filtered_clusters):
            print(f'Cluster #{i}: avg_distance: {c.get("avg_distance"):.3f} ================================================\n')
            for f in c.get('functions'):
                print(indent(f'{f.get("file")}:{f.get("line")}', '    '))
                print(indent(f.get('text'), '    ') + '\n')
