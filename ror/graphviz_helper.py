from typing import List
import logging
import graphviz


def draw_rank(alternatives: List[str], suffix='') -> str:
    dot = graphviz.Digraph(comment='ROR result', graph_attr={'dpi': '300'})
    format = 'jpg'
    dot.format = format
    last_node_id = 1
    for rank_variable in alternatives:
        if type(rank_variable) in [list, tuple]:
            dot.node(str(last_node_id), ', '.join(rank_variable))
        else:
            dot.node(str(last_node_id), rank_variable)
        if last_node_id > 1:
            dot.edge(str(last_node_id-1), str(last_node_id))
        last_node_id += 1

    # save graph
    end = '' if len(suffix) < 1 else f'_{suffix}'

    from os import path
    current_dir = path.abspath(path.curdir)
    filename = path.join(current_dir, f'output/result{end}')
    rendered_filename = dot.render(filename, view=False)
    logging.info(f'Saving final rank to {filename}')
    return rendered_filename
