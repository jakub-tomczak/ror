from ror.Relation import PREFERENCE, Relation
import unittest
from ror.Constraint import ConstraintVariable, Constraint, ValueConstraintVariable, merge_constraints, merge_variables, ConstraintVariablesSet


class TestConstraintClass(unittest.TestCase):
    def test_constraint_variable_equals(self):
        variable_constr_1 = ConstraintVariable('variable1', 0.0)
        variable_constr_1_1 = ConstraintVariable('variable1', 0.0)
        variable_constr_2 = ConstraintVariable('variable2', 0.0)

        self.assertEqual(variable_constr_1, variable_constr_1_1)
        self.assertNotEqual(variable_constr_1, variable_constr_2)

        value_variable_constr_1 = ValueConstraintVariable(0.0)
        value_variable_constr_2 = ValueConstraintVariable(0.0)

        self.assertEqual(value_variable_constr_1, value_variable_constr_2)

    def test_constraint_variables_set(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_1_1 = ConstraintVariable('variable1', 1.0)
        variable_constr_2 = ConstraintVariable('variable2', 4.0)
        value_variable_constr_1 = ValueConstraintVariable(2.0)

        constraint_variables_set = ConstraintVariablesSet([variable_constr_1])

        constraint_variables_set.add_variable(variable_constr_1_1)
        constraint_variables_set.add_variable(variable_constr_2)
        constraint_variables_set.add_variable(value_variable_constr_1)

        self.assertEqual(len(constraint_variables_set.variables), 3)
        self.assertAlmostEqual(
            constraint_variables_set[ValueConstraintVariable.name].coefficient, 2.0)
        self.assertAlmostEqual(
            constraint_variables_set['variable1'].coefficient, 3.0)
        self.assertAlmostEqual(
            constraint_variables_set['variable2'].coefficient, 4.0)

    def test_constraint_variables_set_multiplication(self):
        constraint_variables_set = ConstraintVariablesSet([
            ConstraintVariable('variable1', 2.0),
            ConstraintVariable('variable2', 3.0),
            ValueConstraintVariable(5.0)
        ])

        constraint_variables_set.multiply_by_scalar(-10.0);

        self.assertEqual(len(constraint_variables_set.variables), 3)
        self.assertAlmostEqual(
            constraint_variables_set[ValueConstraintVariable.name].coefficient, -50.0)
        self.assertAlmostEqual(
            constraint_variables_set['variable1'].coefficient, -20.0)
        self.assertAlmostEqual(
            constraint_variables_set['variable2'].coefficient, -30.0)

    def test_constraint_creation(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_2 = ConstraintVariable('variable2', 3.0)
        variable_constr_3 = ConstraintVariable('variable2', 2.0)

        value_variable_constr_1 = ValueConstraintVariable(2.0)
        value_variable_constr_2 = ValueConstraintVariable(1.0)

        variables = ConstraintVariablesSet([
            variable_constr_1,
            variable_constr_2,
            variable_constr_3,
            value_variable_constr_1,
            value_variable_constr_2
        ])

        constraint = Constraint(variables, Relation('some relation', '<='), 'test_constraint')

        self.assertEqual(constraint.number_of_variables, 2)
        self.assertAlmostEqual(constraint.get_variable(
            'variable1').coefficient, 2.0, places=8)
        self.assertAlmostEqual(constraint.get_variable(
            'variable2').coefficient, 5.0, places=8)
        self.assertAlmostEqual(
            constraint.free_variable.coefficient, 3.0, places=8)

        self.assertEqual(constraint.name, 'test_constraint')
        self.assertEqual(constraint.relation.sign, '<=')

    def test_constraint_creation_without_variables(self):
        constraint = Constraint(ConstraintVariablesSet(), Relation('some relation', '<='), 'test_constraint')

        self.assertEqual(constraint.number_of_variables, 0)
        self.assertEqual(constraint.free_variable.coefficient, 0.0)

        self.assertEqual(constraint.name, 'test_constraint')
        self.assertEqual(constraint.relation.sign, '<=')

    def test_adding_variables(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)

        variables = ConstraintVariablesSet([
            variable_constr_1
        ])

        constraint = Constraint(variables, Relation('some relation', '<='), 'test_constraint')

        constraint.add_variable(ValueConstraintVariable(1.0))
        constraint.add_variable(ConstraintVariable('variable1', 1.0))
        constraint.add_variable(ConstraintVariable('variable1', 3.0))
        constraint.add_variable(ConstraintVariable('variable2', 5.0))

        self.assertAlmostEqual(constraint.get_variable(
            'variable1').coefficient, 6.0, places=8)
        self.assertAlmostEqual(constraint.get_variable(
            'variable2').coefficient, 5.0, places=8)
        self.assertAlmostEqual(
            constraint.free_variable.coefficient, 1.0, places=8)

    def test_constraint_multiplication_doesnt_change_relation(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_2 = ConstraintVariable('variable2', 3.0)
        free_variable = ValueConstraintVariable(5.0)

        variables = ConstraintVariablesSet([
            variable_constr_1,
            variable_constr_2,
            free_variable
        ])

        constraint = Constraint(variables, Relation('some relation', '>='), 'test_constraint')
        constraint.multiply_by_scalar(2.0)
        
        self.assertAlmostEqual(constraint.get_variable(
            'variable1').coefficient, 2.0*(2.0), places=8)
        self.assertAlmostEqual(constraint.get_variable(
            'variable2').coefficient, 3.0*(2.0), places=8)
        self.assertAlmostEqual(
            constraint.free_variable.coefficient, 5.0*(2.0), places=8)
        self.assertEqual(constraint.relation.sign, '>=')

    def test_constraint_multiplication_changes_relation(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_2 = ConstraintVariable('variable2', 3.0)
        free_variable = ValueConstraintVariable(5.0)

        variables = ConstraintVariablesSet([
            variable_constr_1,
            variable_constr_2,
            free_variable
        ])

        constraint = Constraint(variables, Relation('some relation', '>='), 'test_constraint')
        constraint.multiply_by_scalar(-2.0)
        
        self.assertAlmostEqual(constraint.get_variable(
            'variable1').coefficient, 2.0*(-2.0), places=8)
        self.assertAlmostEqual(constraint.get_variable(
            'variable2').coefficient, 3.0*(-2.0), places=8)
        self.assertAlmostEqual(
            constraint.free_variable.coefficient, 5.0*(-2.0), places=8)
        self.assertEqual(constraint.relation.sign, '<=')

    def test_merging_constraints(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_2 = ConstraintVariable('variable2', 3.0)
        variable_constr_3 = ConstraintVariable('variable1', 10.0)
        variable_constr_4 = ConstraintVariable('variable2', 20.0)

        value_variable_constr_1 = ValueConstraintVariable(2.0)
        value_variable_constr_2 = ValueConstraintVariable(4.0)

        constraint1 = Constraint(ConstraintVariablesSet([
            variable_constr_1,
            variable_constr_2,
            value_variable_constr_1
        ]),
            PREFERENCE,
            "constr1"
        )
        constraint2 = Constraint(ConstraintVariablesSet(
            [
                variable_constr_3,
                variable_constr_4,
                value_variable_constr_2
            ]),
            PREFERENCE,
            "constr2"
        )

        merged_constraint = merge_constraints([constraint1, constraint2])

        self.assertEqual(merged_constraint.name,
                         f'merged_constraint_constr1_constr2')
        self.assertEqual(len(merged_constraint.variables), 2)

        self.assertAlmostEqual(merged_constraint.get_variable(
            'variable1').coefficient, 12.0, places=8)
        self.assertAlmostEqual(merged_constraint.get_variable(
            'variable2').coefficient, 23.0, places=8)
        self.assertAlmostEqual(
            merged_constraint.free_variable.coefficient, 6.0, places=8)

    def test_merging_different_variables(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_2 = ConstraintVariable('variable2', 3.0)

        merged_not_the_same_variables = merge_variables(
            variable_constr_1, [variable_constr_2])
        self.assertEqual(len(merged_not_the_same_variables), 2)
        self.assertTrue(variable_constr_1 in merged_not_the_same_variables)
        self.assertEqual(merged_not_the_same_variables[0].name, 'variable2')
        self.assertEqual(merged_not_the_same_variables[0].coefficient, 3.0)
        self.assertEqual(merged_not_the_same_variables[1].name, 'variable1')
        self.assertEqual(merged_not_the_same_variables[1].coefficient, 2.0)

    def test_merging_the_same_variables(self):
        variable_constr_1 = ConstraintVariable('variable1', 2.0)
        variable_constr_2 = ConstraintVariable('variable2', 3.0)
        variable_constr_3 = ConstraintVariable('variable1', 10.0)

        merged_not_the_same_variables = merge_variables(
            variable_constr_1,
            [variable_constr_2, variable_constr_3]
        )
        self.assertEqual(len(merged_not_the_same_variables), 2)
        self.assertEqual(merged_not_the_same_variables[0].name, 'variable2')
        self.assertEqual(merged_not_the_same_variables[0].coefficient, 3.0)
        self.assertEqual(merged_not_the_same_variables[1].name, 'variable1')
        self.assertEqual(merged_not_the_same_variables[1].coefficient, 12.0)
