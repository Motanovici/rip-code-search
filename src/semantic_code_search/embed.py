import gzip
import os
import sys
import pickle
from textwrap import dedent

import numpy as np
from tree_sitter import Tree
from tree_sitter_languages import get_parser
from tqdm import tqdm
import linecache

def _supported_file_extensions():
    return {
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.py': 'python',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.kt': 'kotlin',
        '.kts': 'kotlin',
        '.ktm': 'kotlin',
        '.php': 'php',
    }


def _traverse_tree(tree):
    cursor = tree.walk()
    while cursor.node:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return


def _extract_functions(nodes, fp, file_content, relevant_node_types):
    out = []
    file_lines = file_content.split('\n')
    for n in nodes:
        if n.type in relevant_node_types:
            start_line, end_line = n.start_point[0], n.end_point[0] + 1
            node_text = dedent('\n'.join(file_lines[start_line:end_line]))
            out.append({'file': fp, 'line': start_line, 'text': node_text})
    return out

import linecache

def _get_repo_functions(root, supported_file_extensions, relevant_node_types):
    functions = []
    print('Extracting functions from {}'.format(root))
    for fp in tqdm([root + '/' + f for f in os.popen('git -C {} ls-files'.format(root)).read().split('\n')]):
        if not os.path.isfile(fp):
            continue
        with open(fp, 'r') as f:
            lang = supported_file_extensions.get(fp[fp.rfind('.'):])
            if lang:
                parser = get_parser(lang)
                file_content = f.read()
                tree = parser.parse(bytes(file_content, 'utf8'))
                all_nodes = list(_traverse_tree(tree.root_node))
                functions.extend(_extract_functions(
                    all_nodes, fp, file_content, relevant_node_types))
    return functions


def do_embed(args, model):
    nodes_to_extract = ['function_definition', 'method_definition', 'function_declaration', 'method_declaration']
    functions = _get_repo_functions(args.path_to_repo, _supported_file_extensions(), nodes_to_extract)
    num_functions = len(functions)
    if not functions:
        raise ValueError(f"No supported languages found in {args.path_to_repo}. Exiting")
    print(f"Embedding {num_functions} functions in {int(np.ceil(num_functions / args.batch_size))} batches. This is done once and cached in .embeddings")
    corpus_embeddings = model.encode([f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)
    dataset = {
        'functions': functions,
        'embeddings': corpus_embeddings,
        'model_name': args.model_name_or_path
    }
    with gzip.open(args.path_to_repo + '/.embeddings', 'wb', compresslevel=6) as f:
        pickle.dump(dataset, f)
    return dataset
