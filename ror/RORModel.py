from ror.Constraint import Constraint
from ror.Relation import INDIFFERENCE, Relation
from ror.constraints_constants import ConstraintsName
from ror.slope_constraints import create_slope_constraints
from ror.min_max_value_constraints import create_max_value_constraint, create_min_value_constraints
from ror.monotonicity_constraints import create_monotonicity_constraints
from ror.Model import Model
from ror.inner_maximization_constraints import create_inner_maximization_constraints
from ror.Dataset import RORDataset
from typing import List


class RORModel(Model):
    def __init__(self, dataset: RORDataset, alpha: float, name: str, step: int = 1):
        super().__init__([], name)
        assert dataset is not None, "Dataset must not be None"
        self._dataset = dataset
        self._alpha = alpha

        # preferences
        prefernce_constraints = [preference.to_constraint(
            self._dataset, self._alpha) for preference in dataset.preferenceRelations]
        self.add_constraints(prefernce_constraints, ConstraintsName.PREFERENCE_INFORMATION.value)
        prefernce_intensity_constraints = [preference.to_constraint(
            self._dataset, self._alpha) for preference in dataset.intensityRelations]
        self.add_constraints(prefernce_intensity_constraints, ConstraintsName.PREFERENCE_INTENSITY_INFORMATION.value)

        # monotonicity
        monotonicity_constraints = create_monotonicity_constraints(
            self._dataset)
        for criterion in monotonicity_constraints:
            self.add_constraints(monotonicity_constraints[criterion], ConstraintsName.monotonicity(criterion))

        # min-max
        min_constraints = create_min_value_constraints(self._dataset)
        self.add_constraints(min_constraints, ConstraintsName.MIN_CONSTRAINTS.value)
        max_constraint = create_max_value_constraint(self._dataset)
        self.add_constraint(max_constraint, ConstraintsName.MAX_CONSTRAINTS.value)

        # inner maximization
        inner_maximization_constraints = create_inner_maximization_constraints(
            self._dataset)
        self.add_constraints(inner_maximization_constraints, ConstraintsName.INNER_MAXIMIZATION.value)

        # slope
        slope_constraints: List[Constraint] = None
        if step == 2:
            slope_constraints = create_slope_constraints(self._dataset, Relation('=='))
        else:
            slope_constraints = create_slope_constraints(self._dataset)
        self.add_constraints(slope_constraints, ConstraintsName.SLOPE.value)

    @property
    def dataset(self) -> RORDataset:
        return self._dataset
