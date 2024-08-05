from points import Points


class Stations:
    def __init__(self, name: str, point: Points,
                 instr_orientation_flag: bool = False,
                 instr_orientation_rmse: float = None):
        self._name = name
        self._point = point
        self._instr_orientation_flag = instr_orientation_flag
        self._instr_orientation_rmse = instr_orientation_rmse

    @property
    def name(self) -> str:
        return self._name

    @property
    def point(self) -> Points:
        return self._point

    @property
    def instr_orientation_flag(self) -> bool:
        return self._instr_orientation_flag

    @property
    def instr_orientation_rmse(self) -> float:
        return self._instr_orientation_rmse

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @point.setter
    def point(self, new_point: Points):
        self._point = new_point

    @instr_orientation_flag.setter
    def instr_orientation_flag(self, new_instr_orientation_flag: bool):
        self._instr_orientation_flag = new_instr_orientation_flag

    @instr_orientation_rmse.setter
    def instr_orientation_rmse(self, new_instr_orientation_rmse: float):
        self._instr_orientation_rmse = new_instr_orientation_rmse
