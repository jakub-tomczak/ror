class Relation:
    possible_relations = [">=", "<=", "==", ">", "<"]
    def __init__(self, sign: str, name: str = None):
        assert sign in Relation.possible_relations,\
            f"Invalid sign {sign}, accepted values are: {','.join(Relation.possible_relations)}"
        self._sign = sign
        self._name = name

    def __repr__(self) -> str:
        return f'<Relation [name: {self._name}, sign: {self._sign}]>'

    def __hash__(self) -> int:
        return hash(self.__attributes)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Relation) and self.__attributes == other.__attributes

    @property
    def __attributes(self):
        return (self._name, self._sign)

    @property
    def sign(self) -> str:
        return self._sign

    @property
    def name(self) -> str:
        return self._name
    
    @sign.setter
    def sign(self, value: str):
        self._sign = value

WEAK_PREFERENCE = Relation('<=', 'weak preference')
PREFERENCE = Relation('<=', 'preference')
INDIFFERENCE = Relation('==', 'indifference')

PREFERENCE_NAME_TO_RELATION = {
    "preference": PREFERENCE,
    "weak preference": WEAK_PREFERENCE,
    "indifference": INDIFFERENCE
}