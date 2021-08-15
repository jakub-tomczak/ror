from ror.slope_constraints import create_slope_constraints
from ror.min_max_value_constraints import create_max_value_constraint, create_min_value_constraints
from ror.monotonicity_constraints import create_monotonicity_constraints
from ror.Model import Model
from ror.inner_maximization_constraints import create_inner_maximization_constraints
from ror.Dataset import RORDataset


class RORModel(Model):
    def __init__(self, dataset: RORDataset, alpha: float, notes: str):
        super().__init__([], notes)
        assert dataset is not None, "Dataset must not be None"
        self._dataset = dataset
        self._alpha = alpha

        # preferences
        prefernce_constraints = [preference.to_constraint(
            self._dataset, self._alpha) for preference in dataset.preferenceRelations]
        self.add_constraints(prefernce_constraints)
        prefernce_intensity_constraints = [preference.to_constraint(
            self._dataset, self._alpha) for preference in dataset.intensityRelations]
        self.add_constraints(prefernce_intensity_constraints)

        # monotonicity
        monotonicity_constraints = create_monotonicity_constraints(
            self._dataset)
        for criterion in monotonicity_constraints:
            self.add_constraints(monotonicity_constraints[criterion])

        # min-max
        min_constraints = create_min_value_constraints(self._dataset)
        self.add_constraints(min_constraints)
        max_constraint = create_max_value_constraint(self._dataset)
        self.add_constraint(max_constraint)

        # inner maximization
        inner_maximization_constraints = create_inner_maximization_constraints(
            self._dataset)
        self.add_constraints(inner_maximization_constraints)

        # slope
        slope_constraints = create_slope_constraints(self._dataset)
        self.add_constraints(slope_constraints)
