import pandas as pd
from typing import Dict, List, Union
from ror.OptimizationResult import OptimizationResult
from ror.Constraint import ConstraintVariablesSet, ConstraintVariable
from ror.data_loader import read_dataset_from_txt
from ror.RORModel import RORModel
from ror.d_function import d

data = read_dataset_from_txt("tests/datasets/ror_full_dataset.txt")
# step 1
print('Starting step 1')
alpha = 0.0
model = RORModel(data, alpha, f"ROR Model, step 1, with alpha {alpha}")
model.target = ConstraintVariablesSet([
    ConstraintVariable("delta", 1.0)
])
result = model.solve()
print("Solved step 1, delta value is", result.objective_value)

print('Starting step 2')
# assign delta value to the data
data.delta = result.objective_value

alpha_range = [0.0, 0.5, 0.75, 1.0]
alpha_value_key_generator = lambda alpha: f"alpha_{alpha}"
# results => Dict with columns for all results with d function values calculated for each alternative
# and one column with alternative names
results: Dict[str, Union[str, List[OptimizationResult]]] = {alpha_value_key_generator(alpha): [] for alpha in alpha_range}
results["id"] = []

for alternative in data.alternatives:
    for alpha in alpha_range:
        model = RORModel(data, alpha, f"ROR Model, step 2, with alpha {alpha}, alternative {alternative}")
        model.target = d(alternative, alpha, data)
        result = model.solve()
        results[alpha_value_key_generator(alpha)].append(result.objective_value)
        print(f"alternative {alternative}, objective value {result.objective_value}")
    results["id"].append(alternative)


all_data = pd.DataFrame(results)
all_data.set_index("id", inplace=True)
# create series with sum of all alphas
sum_per_alternative_series = sum([all_data[alpha_value_key_generator(alpha)] for alpha in alpha_range])
sum_per_alternative_series.name = "sum"
all_data = pd.concat([all_data, sum_per_alternative_series], axis=1)

# display result
print(all_data)
