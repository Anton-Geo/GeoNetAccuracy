class Instruments:
    def __init__(self, type_name: str, instrument_name: str, inventory_number: str = None,
                 linear_random_error: float = None, ppm: float = None,
                 angular_random_error: float = None):
        self._type_name = type_name
        self._instrument_name = instrument_name
        self._inventory_number = inventory_number
        self._linear_random_error = linear_random_error
        self._ppm = ppm
        self._angular_random_error = angular_random_error

    @property
    def type_name(self) -> str:
        return self._type_name

    @type_name.setter
    def type_name(self, new_type_name: str):
        self._type_name = new_type_name

    @property
    def instrument_name(self) -> str:
        return self._instrument_name

    @instrument_name.setter
    def instrument_name(self, new_instrument_name: str):
        self._instrument_name = new_instrument_name

    @property
    def inventory_number(self) -> str:
        return self._inventory_number

    @inventory_number.setter
    def inventory_number(self, new_inventory_number: str):
        self._inventory_number = new_inventory_number

    @property
    def linear_random_error(self) -> float:
        return self._linear_random_error

    @linear_random_error.setter
    def linear_random_error(self, new_linear_random_error: float):
        self._linear_random_error = new_linear_random_error

    @property
    def ppm(self) -> float:
        return self._ppm

    @ppm.setter
    def ppm(self, new_ppm: float):
        self._ppm = new_ppm

    @property
    def angular_random_error(self) -> float:
        return self._angular_random_error

    @angular_random_error.setter
    def angular_random_error(self, new_angular_random_error: float):
        self._angular_random_error = new_angular_random_error


class GnssReceivers(Instruments):
    def __init__(self, type_name: str, instrument_name: str, inventory_number: str,
                 linear_random_error: float, ppm: float):
        super().__init__(type_name=type_name,
                         instrument_name=instrument_name,
                         inventory_number=inventory_number)
        self._linear_random_error = linear_random_error
        self._ppm = ppm

    @property
    def linear_random_error(self) -> float:
        return self._linear_random_error

    @linear_random_error.setter
    def linear_random_error(self, new_linear_random_error: float):
        self._linear_random_error = new_linear_random_error

    @property
    def ppm(self) -> float:
        return self._ppm

    @ppm.setter
    def ppm(self, new_ppm: float):
        self._ppm = new_ppm


class Theodolites(Instruments):
    def __init__(self, type_name: str, instrument_name: str, inventory_number: str,
                 angular_random_error: float):
        super().__init__(type_name=type_name,
                         instrument_name=instrument_name,
                         inventory_number=inventory_number)
        self._angular_random_error = angular_random_error

    @property
    def angular_random_error(self) -> float:
        return self._angular_random_error

    @angular_random_error.setter
    def angular_random_error(self, new_angular_random_error: float):
        self._angular_random_error = new_angular_random_error


class TotalStations(Instruments):
    def __init__(self, type_name: str, instrument_name: str, inventory_number: str,
                 linear_random_error: float, ppm: float, angular_random_error: float):
        super().__init__(type_name=type_name,
                         instrument_name=instrument_name,
                         inventory_number=inventory_number)
        self._linear_random_error = linear_random_error
        self._ppm = ppm
        self._angular_random_error = angular_random_error

    @property
    def linear_random_error(self) -> float:
        return self._linear_random_error

    @linear_random_error.setter
    def linear_random_error(self, new_linear_random_error: float):
        self._linear_random_error = new_linear_random_error

    @property
    def ppm(self) -> float:
        return self._ppm

    @ppm.setter
    def ppm(self, new_ppm: float):
        self._ppm = new_ppm

    @property
    def angular_random_error(self) -> float:
        return self._angular_random_error

    @angular_random_error.setter
    def angular_random_error(self, new_angular_random_error: float):
        self._angular_random_error = new_angular_random_error
