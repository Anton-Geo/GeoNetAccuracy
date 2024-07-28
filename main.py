"""
This is the main file to calculate preliminary accuracy assessment
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple  # , Any,


PATH_IMPORT_EXCEL = 'Data_for_preliminary_aa.xlsx'


class Points:
    def __init__(self, name: str, x: float, y: float,
                 rmse_x: float = 0.0, rmse_y: float = 0.0, cov_xy: float = 0.0):
        self._name = name
        self._x = x
        self._y = y
        self._rmse_x = rmse_x
        self._rmse_y = rmse_y
        self._cov_xy = cov_xy

    @property
    def name(self) -> str:
        return self._name

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def rmse_x(self) -> float:
        return self._rmse_x

    @property
    def rmse_y(self) -> float:
        return self._rmse_y

    @property
    def cov_xy(self) -> float:
        return self._cov_xy

    @name.setter
    def name(self, new_name: float):
        self._name = new_name

    @x.setter
    def x(self, new_x: float):
        self._x = new_x

    @y.setter
    def y(self, new_y: float):
        self._y = new_y

    @rmse_x.setter
    def rmse_x(self, new_rmse_x: float):
        self._rmse_x = new_rmse_x

    @rmse_y.setter
    def rmse_y(self, new_rmse_y: float):
        self._rmse_y = new_rmse_y

    @cov_xy.setter
    def cov_xy(self, new_cov_xy: float):
        self._cov_xy = new_cov_xy


class BasePoints(Points):
    def __init__(self, name: str, x: float, y: float):
        super().__init__(name, x, y)


class RefinedPoints(Points):
    def __init__(self, name: str, x: float, y: float,
                 rmse_x: float, rmse_y: float, cov_xy: float):
        super().__init__(name, x, y, rmse_x, rmse_y, cov_xy)


class EstimatedPoints(Points):
    def __init__(self, name: str, x: float, y: float):
        super().__init__(name, x, y)


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


def import_excel_data(path: str = PATH_IMPORT_EXCEL,
                      instruments_sheet_name: str = 'Instruments',
                      points_sheet_name: str = 'Points',
                      measurements_sheet_name: str = 'Measurements'):
    # Load data from Excel file
    df_instruments = pd.read_excel(path, sheet_name=instruments_sheet_name)
    df_points = pd.read_excel(path, sheet_name=points_sheet_name)
    df_measurements = pd.read_excel(path, sheet_name=measurements_sheet_name)

    # print('---- df_instruments ----')
    # print(df_instruments.head(10))
    # print('---- df_points ----')
    # print(df_points.head(10))
    # print('---- df_measurements ----')
    # print(df_measurements.head(10))
    # print('-- -- -- -- -- -- -- -- --')

    # Go through the lines of df_instruments and create the corresponding instrument objects
    list_instruments = []
    for index, row in df_instruments.iterrows():
        if row['instrument_type'] == 'gnss_receiver':
            gnss_receiver = GnssReceivers(row['instrument_type'], row['instrument_name'],
                                          row['inventory_number'],
                                          row['linear_rmse'], row['linear_ppm'])
            list_instruments.append(gnss_receiver)
        elif row['instrument_type'] == 'theodolite':
            theodolite = Theodolites(row['instrument_type'], row['instrument_name'],
                                     row['inventory_number'],
                                     row['hor_angle_rmse'])
            list_instruments.append(theodolite)
        elif row['instrument_type'] == 'total_station':
            total_station = TotalStations(row['instrument_type'], row['instrument_name'],
                                          row['inventory_number'],
                                          row['linear_rmse'], row['linear_ppm'],
                                          row['hor_angle_rmse'])
            list_instruments.append(total_station)

    print('---- instruments ----')
    print(*[(i.inventory_number, i.type_name, i.instrument_name,
             i.linear_random_error, i.ppm,
             i.angular_random_error) for i in list_instruments], sep='\n')

    # Go through the lines of df_points and create the corresponding point objects
    list_points = []
    for index, row in df_points.iterrows():
        if row['point_type'] == 'base':
            base_point = BasePoints(row['point_name'], row['x'], row['y'])
            list_points.append(base_point)
        elif row['point_type'] == 'estimated':
            estimated_point = EstimatedPoints(row['point_name'], row['x'], row['y'])
            list_points.append(estimated_point)
        elif row['point_type'] == 'refined':
            if pd.notna(row['rmse_x']) and pd.notna(row['rmse_y']) and pd.notna(row['cov_xy']):
                refined_point = RefinedPoints(row['point_name'], row['x'], row['y'],
                                              row['rmse_x'], row['rmse_y'], row['cov_xy'])
            elif pd.notna(row['rmse_x']) and pd.notna(row['rmse_y']):
                refined_point = RefinedPoints(row['point_name'], row['x'], row['y'],
                                              row['rmse_x'], row['rmse_y'], cov_xy=0.0)
            else:
                refined_point = RefinedPoints(row['point_name'], row['x'], row['y'],
                                              rmse_x=0.0, rmse_y=0.0, cov_xy=0.0)
            list_points.append(refined_point)

    # Go through the lines of df_measurements and create list of Stations
    seen_station_points = set()
    list_stations = []
    for index, row in df_measurements.iterrows():
        station_name = row['station_name']
        start_point_name = row['station_point_name']
        start_point = next((point for point in list_points if point.name == start_point_name),
                           None)
        if row['measurement_type'] == 'angular':
            orientation_flag = True
        else:
            orientation_flag = False

        if station_name not in seen_station_points:
            station = Stations(station_name, start_point, orientation_flag)
            list_stations.append(station)
            seen_station_points.add(station.name)

        if station_name in seen_station_points and row['measurement_type'] == 'angular':
            station = next((st for st in list_stations if st.name == station_name), None)
            if station and station.instr_orientation_flag is False:
                station.instr_orientation_flag = True

        # print([(station.name, station.point.name, station.instr_orientation_flag) for station in
        #        list_stations])

    # Go through the lines of df_measurements and create the corresponding measurement objects
    list_measurements = []
    for index, row in df_measurements.iterrows():
        if row['measurement_type'] in ('linear', 'angular'):
            instrument_id = row['instrument_inventory_number']
            station_name = row['station_name']
            start_point_name = row['station_point_name']
            end_point_name = row['aim_point_name']

            # Find the corresponding instrument in the list_instruments
            instrument = next((instrument for instrument in list_instruments
                               if instrument.inventory_number == instrument_id), None)

            # Find the corresponding station in the list_stations
            station = next((st for st in list_stations if st.name == station_name), None)

            # Find the corresponding points in the list_points
            start_point = next((point for point in list_points if point.name == start_point_name),
                               None)
            end_point = next((point for point in list_points if point.name == end_point_name),
                             None)

            if instrument and station and start_point and end_point:
                if row['measurement_type'] == 'linear':
                    measurement = LinearMeasurements(station, start_point, end_point, instrument)
                    list_measurements.append(measurement)
                elif row['measurement_type'] == 'angular':
                    measurement = AngularMeasurements(station, start_point, end_point, instrument)
                    list_measurements.append(measurement)

            else:
                print(f"Warning: Could not find instrument or station or points for measurement "
                      f"instrument {instrument}, station {station_name}: "
                      f"{start_point_name} to {end_point_name}")

    print('---- measurements ----')
    print(*[(m.instrument.instrument_name,
             m.station.name, m.start_point.name, m.end_point.name,
             m.calculate_rmse()) for m in list_measurements], sep='\n')

    return list_points, list_measurements, list_stations


def create_filtered_points(list_of_points: List[Points],
                           all_measurements: List[Measurements]) -> Tuple[Dict, Dict]:

    seen_points = set()
    dict_estimated_points = {}
    k = 0
    for e_point in list_of_points:
        if isinstance(e_point, EstimatedPoints) and e_point not in seen_points:
            dict_estimated_points[k] = e_point
            seen_points.add(e_point)
            k += 1

    dict_refined_points = {}
    k = 0
    for r_point in list_of_points:
        if isinstance(r_point, RefinedPoints) and r_point not in seen_points:
            dict_refined_points[k] = r_point
            seen_points.add(r_point)
            k += 1

    return dict_estimated_points, dict_refined_points


def create_filtered_measurements(filtered_points: Tuple[Dict, Dict],
                                 all_measurements: List[Union[LinearMeasurements,
                                                              AngularMeasurements]]
                                 ) -> List[Union[LinearMeasurements,
                                                 AngularMeasurements]]:
    filtered_measurements = []
    dict_estimated_points, dict_refined_points = filtered_points

    # Create a set of points from the values of the filtered_points dictionary
    filtered_points_set = set(dict_estimated_points.values()) | set(dict_refined_points.values())
    for measurement in all_measurements:
        if isinstance(measurement, LinearMeasurements) and \
                (measurement.start_point in filtered_points_set or
                 measurement.end_point in filtered_points_set):
            filtered_measurements.append(measurement)
        elif isinstance(measurement, AngularMeasurements):
            filtered_measurements.append(measurement)

    return filtered_measurements


def create_jacobian_matrix(filtered_points: Tuple[Dict, Dict],
                           filtered_measurements: List[Union[LinearMeasurements,
                                                             AngularMeasurements]],
                           list_stations: List) -> np.ndarray:
    dict_estimated_points, dict_refined_points = filtered_points
    num_measurements = len(filtered_measurements)
    num_orients = sum(station.instr_orientation_flag for station in list_stations)

    num_parameters = len(dict_estimated_points) * 2 + num_orients + len(dict_refined_points) * 2

    matrix_j = np.zeros((num_measurements + len(dict_refined_points) * 2, num_parameters))

    # Filling the matrix according to the parameters of unknown (estimated) points (A_1)
    for i, measurement in enumerate(filtered_measurements):
        for j, point in dict_estimated_points.items():
            if point == measurement.start_point and isinstance(point, EstimatedPoints):
                matrix_j[i, j * 2] = measurement.partial_derivative_sp_x()
                matrix_j[i, j * 2 + 1] = measurement.partial_derivative_sp_y()
            elif point == measurement.end_point and isinstance(point, EstimatedPoints):
                matrix_j[i, j * 2] = measurement.partial_derivative_ep_x()
                matrix_j[i, j * 2 + 1] = measurement.partial_derivative_ep_y()

    # Filling the matrix according to the parameters of stations orientations (A_1)
    stations_with_flag = [station for station in list_stations if station.instr_orientation_flag]
    for i, measurement in enumerate(filtered_measurements):
        for j, station in enumerate(stations_with_flag, start=(len(dict_estimated_points) * 2)):
            if isinstance(measurement, AngularMeasurements) and \
                    measurement.station == station:
                matrix_j[i, j] = measurement.partial_derivative_orientation()

    # Filling the matrix according to the parameters of redefined points (A_2)
    tj = len(dict_estimated_points) * 2 + num_orients
    for i, measurement in enumerate(filtered_measurements):
        for j, point in dict_refined_points.items():
            if point == measurement.start_point and isinstance(point, RefinedPoints):
                matrix_j[i, tj + j * 2] = measurement.partial_derivative_sp_x()
                matrix_j[i, tj + j * 2 + 1] = measurement.partial_derivative_sp_y()
            elif point == measurement.end_point and isinstance(point, RefinedPoints):
                matrix_j[i, tj + j * 2] = measurement.partial_derivative_ep_x()
                matrix_j[i, tj + j * 2 + 1] = measurement.partial_derivative_ep_y()

    # Filling the matrix according to the parameters of redefined points (E)
    ti = len(filtered_measurements)
    tj = len(dict_estimated_points) * 2 + num_orients
    for i, _ in dict_refined_points.items():
        for j, _ in dict_refined_points.items():
            matrix_j[ti + i * 2, tj + j * 2] = 1
            matrix_j[ti + i * 2 + 1, tj + j * 2 + 1] = 1

    with np.printoptions(precision=4, suppress=True):
        print('---- jacobian_matrix ----')
        print(matrix_j)
    return matrix_j


def create_precision_matrix(list_measurements: List[Union[LinearMeasurements,
                                                          AngularMeasurements]],
                            dict_refined_points: Dict) -> np.ndarray:
    num_measurements = len(list_measurements)
    num_refined_points = len(dict_refined_points) * 2
    dimensionality = num_measurements + num_refined_points

    weight_values = []
    # Measurement weights
    for measurement in list_measurements:
        if isinstance(measurement, LinearMeasurements):
            rmse = measurement.calculate_rmse() / 1000
            weight_values.append(1 / rmse**2)
        elif isinstance(measurement, AngularMeasurements):
            rmse = measurement.calculate_rmse()
            weight_values.append(1 / rmse**2)

    weight_matrix = np.zeros((dimensionality, dimensionality))
    np.fill_diagonal(weight_matrix, weight_values)

    # Weights of coordinates of redefined points
    t = len(list_measurements)
    for i, point in dict_refined_points.items():
        rmse_x, rmse_y = point.rmse_x / 1000, point.rmse_y / 1000
        cov_xy = point.cov_xy / 1000
        sub_matrix_k = np.array([[rmse_x**2, cov_xy],
                                 [cov_xy, rmse_y**2]])
        sub_matrix_p = np.linalg.inv(sub_matrix_k)
        weight_matrix[t + i * 2, t + i * 2] = sub_matrix_p[0, 0]
        weight_matrix[t + i * 2 + 1, t + i * 2 + 1] = sub_matrix_p[1, 1]
        weight_matrix[t + i * 2 + 1, t + i * 2] = sub_matrix_p[1, 0]
        weight_matrix[t + i * 2, t + i * 2 + 1] = sub_matrix_p[0, 1]

    with np.printoptions(precision=2, suppress=True):
        print('---- weight_matrix ----')
        print(weight_matrix)
    return weight_matrix


def assess_points_accuracy(list_of_points: List[Points],
                           list_of_measurements: List[Union[LinearMeasurements,
                                                            AngularMeasurements]],
                           list_of_stations: List[Stations]
                           ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:

    filtered_points = create_filtered_points(list_of_points, list_of_measurements)
    dict_estimated_points, dict_refined_points = filtered_points
    filtered_measurements = create_filtered_measurements(filtered_points, list_of_measurements)

    matrix_j = create_jacobian_matrix(filtered_points, filtered_measurements, list_of_stations)  # J
    matrix_p = create_precision_matrix(filtered_measurements, dict_refined_points)  # P
    matrix_k = np.linalg.inv(matrix_j.T @ matrix_p @ matrix_j)  # K

    with np.printoptions(precision=6, suppress=True):
        print('---- covariance_matrix ----')
        print(matrix_k)

    # Writing values to EstimatedPoints instances
    point_table = {}
    for i, point in dict_estimated_points.items():
        if isinstance(point, (RefinedPoints, EstimatedPoints)):
            rmse_x = np.sqrt(matrix_k[i * 2, i * 2])
            rmse_y = np.sqrt(matrix_k[i * 2 + 1, i * 2 + 1])
            cov_xy = matrix_k[i * 2, i * 2 + 1]
            correlation = cov_xy / (rmse_x * rmse_y) if rmse_x * rmse_y != 0 else 0.0

            point.rmse_x = rmse_x
            point.rmse_y = rmse_y
            point.cov_xy = cov_xy

            point_table[point.name] = [point.x, point.y,
                                       round(point.rmse_x * 1000, 2),
                                       round(point.rmse_y * 1000, 2),
                                       round(correlation, 2)]

    # Writing values to Station instances
    stations_with_flag = [station for station in list_of_stations if station.instr_orientation_flag]
    station_table = {}
    for i, station in enumerate(stations_with_flag, start=(len(dict_estimated_points) * 2)):
        instr_orientation_rmse = np.sqrt(matrix_k[i, i])

        station.instr_orientation_rmse = instr_orientation_rmse

        station_table[station.name] = [station.point.name, station.point.x, station.point.y,
                                       round(station.instr_orientation_rmse, 2)]

    # Writing values to RefinedPoints instances
    t = len(dict_estimated_points) * 2 + len(stations_with_flag)
    for i, point in dict_refined_points.items():
        if isinstance(point, (RefinedPoints, EstimatedPoints)):
            rmse_x = np.sqrt(matrix_k[t + i * 2, t + i * 2])
            rmse_y = np.sqrt(matrix_k[t + i * 2 + 1, t + i * 2 + 1])
            cov_xy = matrix_k[t + i * 2, t + i * 2 + 1]
            correlation = cov_xy / (rmse_x * rmse_y) if rmse_x * rmse_y != 0 else 0.0

            point.rmse_x = rmse_x
            point.rmse_y = rmse_y
            point.cov_xy = cov_xy

            point_table[point.name] = [point.x, point.y,
                                       round(point.rmse_x * 1000, 2),
                                       round(point.rmse_y * 1000, 2),
                                       round(correlation, 2)]

    return point_table, station_table


def main():
    result_import = assess_points_accuracy(*import_excel_data())
    print('---- Network points accuracy assessment ----')
    print(*result_import[0].items(), sep='\n')
    print('---- Station orientation accuracy assessment ----')
    print(*result_import[1].items(), sep='\n')


if __name__ == '__main__':
    main()
