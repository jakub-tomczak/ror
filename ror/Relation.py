class Relation:
    possible_relations = [">=", "<=", "==", ">", "<"]
    def __init__(self, sign: str, name: str = None):
        assert sign in Relation.possible_relations,\
            f"Invalid sign {sign}, accepted values are: {','.join(Relation.possible_relations)}"
        self._sign = sign
        self._name = name

    def __repr__(self) -> str:
        return f'<Relation [name: {self._name}, sign: {self._sign}]>'

    @property
    def sign(self):
        return self._sign
    
    @sign.setter
    def sign(self, value: str):
        self._sign = value

WEAK_PREFERENCE = Relation('<=', 'weak preference')
PREFERENCE = Relation('<=', 'preference')
INDIFFERENCE = Relation('==', 'indifference')