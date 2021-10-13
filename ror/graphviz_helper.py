from typing import List
import logging
import graphviz
import datetime

def get_date_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H-%M-%S")

def draw_rank(alternatives: List[str], filename: str) -> str:
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

    from os import path
    current_dir = path.abspath(path.curdir)
    filename = path.join(current_dir, 'output', get_date_time(), filename)
    rendered_filename = dot.render(filename, view=False)
    logging.info(f'Saving final rank to {filename}')
    return rendered_filename
