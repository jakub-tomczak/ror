from typing import Dict
import pylatex as tex
from pylatex.utils import NoEscape
from ror.Constraint import ConstraintVariablesSet
import logging


def _generate_latex_model(model: 'RORModel') -> tex.Document:
    """
    Creates latex document from ROR model.
    """
    geometry_options: Dict[str, str] = {
        "margin": "0.4in",
        "bottom": "0.4in",
    }
    doc = tex.Document(f'ror_model', geometry_options=geometry_options)
    with doc.create(tex.Section(f'Model: {model.name}')):
        doc.append(tex.Subsection('Target:'))
        target: ConstraintVariablesSet = model.target
        first_constraint_positive_added = False
        for variable in target.variables:
            if variable.coefficient == 0:
                continue
            if variable.coefficient > 0 and first_constraint_positive_added:
                doc.append(NoEscape(f'$+ {variable.coefficient} * {variable.name}$ \ '))
            else:
                doc.append(NoEscape(f'${variable.coefficient} * {variable.name}$ \ '))
                first_constraint_positive_added = True
        doc.append(tex.Subsection('Constraints'))
        for name in model.constraints_dict:
            doc.append(tex.Subsubsection(f'{name} constraints'))
            for constraint in model.constraints_dict[name]:
                var = constraint.to_latex()
                doc.append(NoEscape(f'${var}$ \\\\'))
    return doc

def export_latex(model: 'RORModel', filename: str) -> str:
    document = _generate_latex_model(model)
    if filename.endswith('.tex'):
        # remove .tex extension (required by pylatex)
        filename = filename[:-len('.tex')]
    document.generate_tex(filename)
    logging.info(f'Exported model to latex file, saved as {filename}')
    return filename
    # doc.generate_pdf('basic_maketitle2', clean_tex=False)
    # tex = doc.dumps()  # The document as string in LaTeX syntax

def export_latex_pdf(model: 'RORModel', filename: str) -> str:
    document = _generate_latex_model(model)
    if filename.endswith('.tex'):
        # remove .pdf extension (required by pylatex)
        filename = filename[:-len('.pdf')]
    document.generate_pdf(filename, clean_tex = True)
    logging.info(f'Exported model to latex pdf file, saved as {filename}')
    return filename
