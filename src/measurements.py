import math
from typing import Union
from points import Points
from instruments import Instruments, GnssReceivers, Theodolites, TotalStations
from stations import Stations


class Measurements:
    def __init__(self, station: Stations,
                 start_point: Points, end_point: Points,
                 instrument: Instruments):
        assert start_point is not end_point, "Start and end points must be different"
        self._station = station
        self._start_point = start_point
        self._end_point = end_point
        self._instrument = instrument

    @property
    def station(self) -> Stations:
        return self._station

    @station.setter
    def station(self, new_station: Points):
        self._station = new_station

    @property
    def start_point(self) -> Points:
        return self._start_point

    @start_point.setter
    def start_point(self, new_start_point: Points):
        self._start_point = new_start_point

    @property
    def end_point(self) -> Points:
        return self._end_point

    @end_point.setter
    def end_point(self, new_end_point: Points):
        self._end_point = new_end_point

    @property
    def instrument(self) -> Instruments:
        return self._instrument

    @instrument.setter
    def instrument(self, new_instrument: Points):
        self._instrument = new_instrument


class LinearMeasurements(Measurements):
    def __init__(self, station: Stations,
                 start_point: Points, end_point: Points,
                 instrument: Union[GnssReceivers, TotalStations]):
        super().__init__(station=station,
                         start_point=start_point, end_point=end_point,
                         instrument=instrument)
        self._random_error = instrument.linear_random_error
        self._ppm = instrument.ppm
        self._value = self.calculate_value()

    @property
    def value(self) -> float:
        return self._value

    def calculate_value(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        delta_y = self.end_point.y - self.start_point.y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return distance

    def calculate_rmse(self) -> float:
        adjusted_rmse = self._random_error + self._ppm * (self._value / 1000)
        return adjusted_rmse

    def partial_derivative_sp_x(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        return -1 * delta_x / self._value

    def partial_derivative_sp_y(self) -> float:
        delta_y = self.end_point.y - self.start_point.y
        return -1 * delta_y / self._value

    def partial_derivative_ep_x(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        return delta_x / self._value

    def partial_derivative_ep_y(self) -> float:
        delta_y = self.end_point.y - self.start_point.y
        return delta_y / self._value


class AngularMeasurements(Measurements):  # as an angular direction!
    RHO = 206265

    def __init__(self, station: Stations,
                 start_point: Points, end_point: Points,
                 instrument: Union[Theodolites, TotalStations]):
        super().__init__(station=station,
                         start_point=start_point, end_point=end_point,
                         instrument=instrument)
        self._random_error = instrument.angular_random_error
        self._value = self.calculate_value()

    @property
    def value(self) -> float:
        return self._value

    def calculate_value(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        delta_y = self.end_point.y - self.start_point.y
        angle = math.degrees(math.atan2(delta_y, delta_x)) * 3600  # in seconds
        return angle

    def calculate_rmse(self) -> float:
        adjusted_rmse = self._random_error
        return adjusted_rmse

    def partial_derivative_sp_x(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        delta_y = self.end_point.y - self.start_point.y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return (delta_y / (distance ** 2)) * self.RHO

    def partial_derivative_sp_y(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        delta_y = self.end_point.y - self.start_point.y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return (-1 * delta_x / (distance ** 2)) * self.RHO

    def partial_derivative_ep_x(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        delta_y = self.end_point.y - self.start_point.y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return (-1 * delta_y / (distance ** 2)) * self.RHO

    def partial_derivative_ep_y(self) -> float:
        delta_x = self.end_point.x - self.start_point.x
        delta_y = self.end_point.y - self.start_point.y
        distance = (delta_x ** 2 + delta_y ** 2) ** 0.5
        return (delta_x / (distance ** 2)) * self.RHO

    def partial_derivative_orientation(self) -> float:
        return 1
