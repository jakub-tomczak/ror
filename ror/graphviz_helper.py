from typing import List
import graphviz


def draw_rank(alternatives: List[str], suffix=''):
    dot = graphviz.Digraph(comment='ROR result')
    dot.format = 'svg'
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

    dot.render(
        f'output/result{end}', view=False)
