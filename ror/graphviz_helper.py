from typing import List
import logging
import graphviz
import os


def draw_rank(alternatives: List[str], dir: str, filename: str) -> str:
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

    filename = os.path.join(dir, filename)
    rendered_filename = dot.render(filename, view=False)
    logging.info(f'Saving final rank to "{filename}"')
    return rendered_filename
