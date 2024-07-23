"""
This is the main file to calculate preliminary accuracy assessment
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union  # , Any, Tuple


PATH_IMPORT_EXCEL = 'Data_for_preliminary_aa.xlsx'


class Points:
    def __init__(self, name: str, x: float, y: float,
                 rmse_x: float = 0.0, rmse_y: float = 0.0, cov_xy: float = 0.0):
        self.name = name
        self._x = x
        self._y = y
        self._rmse_x = rmse_x
        self._rmse_y = rmse_y
        self._cov_xy = cov_xy

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


class Measurements:
    def __init__(self, start_point: Points, end_point: Points):
        assert start_point is not end_point, "Start and end points must be different"
        self._start_point = start_point
        self._end_point = end_point

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


class LinearMeasurements(Measurements):
    def __init__(self, start_point: Points, end_point: Points,
                 random_error: float, ppm: float):
        super().__init__(start_point=start_point, end_point=end_point)
        self._random_error = random_error
        self._ppm = ppm
        self._value = self.calculate_value()

    @property
    def value(self) -> float:
        return self._value

    def calculate_value(self) -> float:
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance

    def calculate_rmse(self) -> float:
        adjusted_rmse = self._random_error + self._ppm * (self._value / 1000)
        return adjusted_rmse

    def partial_derivative_sp_x(self) -> float:
        x_diff = self.end_point.x - self.start_point.x
        return -1 * x_diff / self._value

    def partial_derivative_sp_y(self) -> float:
        y_diff = self.end_point.y - self.start_point.y
        return -1 * y_diff / self._value

    def partial_derivative_ep_x(self) -> float:
        x_diff = self.end_point.x - self.start_point.x
        return x_diff / self._value

    def partial_derivative_ep_y(self) -> float:
        y_diff = self.end_point.y - self.start_point.y
        return y_diff / self._value


def import_points(path: str = PATH_IMPORT_EXCEL,
                  points_sheet_name: str = 'Points',
                  measurements_sheet_name: str = 'Measurements'):
    # Load data from Excel file
    df_points = pd.read_excel(path, sheet_name=points_sheet_name)
    df_measurements = pd.read_excel(path, sheet_name=measurements_sheet_name)

    print(df_points.head())
    print(df_measurements.head())

    list_points = []
    # Go through the lines of df_points and create the corresponding point objects
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

    list_measurements = []
    # Go through the lines of df_measurements and create the corresponding measurement objects
    for index, row in df_measurements.iterrows():
        if row['measurement_type'] == 'linear':
            start_point_name = row['instrument_point_name']
            end_point_name = row['aim_point_name']

            # Find the corresponding points in the list_points
            start_point = next((point for point in list_points if point.name == start_point_name),
                               None)
            end_point = next((point for point in list_points if point.name == end_point_name),
                             None)

            if start_point and end_point:
                measurement = LinearMeasurements(start_point, end_point,
                                                 row['random_error'] / 1000,
                                                 row['ppm'] / 1000)
                list_measurements.append(measurement)
            else:
                print(f"Warning: Could not find points for measurement "
                      f"{start_point_name} to {end_point_name}")

    return list_points, list_measurements


def create_filtered_points(list_of_points: List[Points],
                           all_measurements: List[Measurements]) -> Dict:
    filtered_points = {}
    seen_points = set()

    k = 0
    for point in list_of_points:
        if isinstance(point, (RefinedPoints, EstimatedPoints)) and point not in seen_points:
            filtered_points[k] = point
            seen_points.add(point)
            k += 1

    return filtered_points


def create_filtered_measurements(filtered_points: Dict,
                                 all_measurements: List[Union[LinearMeasurements]]
                                 ) -> List[Union[LinearMeasurements]]:
    filtered_linear_measurements = []

    # Create a set of points from the values of the filtered_points dictionary
    filtered_points_set = set(filtered_points.values())

    for measurement in all_measurements:
        if isinstance(measurement, LinearMeasurements) and \
                (measurement.start_point in filtered_points_set or
                 measurement.end_point in filtered_points_set):
            filtered_linear_measurements.append(measurement)

    return filtered_linear_measurements


def create_jacobian_matrix(filtered_points: Dict,
                           all_measurements: List[Union[LinearMeasurements]]) -> np.ndarray:
    num_measurements = len(all_measurements)
    num_parameters = len(filtered_points) * 2

    matrix = np.zeros((num_measurements, num_parameters))

    for i, measurement in enumerate(all_measurements):
        for j, point in filtered_points.items():
            if point == measurement.start_point:
                matrix[i, j * 2] = measurement.partial_derivative_sp_x()
                matrix[i, j * 2 + 1] = measurement.partial_derivative_sp_y()
            elif point == measurement.end_point:
                matrix[i, j * 2] = measurement.partial_derivative_ep_x()
                matrix[i, j * 2 + 1] = measurement.partial_derivative_ep_y()

    return matrix


def create_precision_matrix(list_measurements: List[Union[LinearMeasurements]]) -> np.ndarray:
    num_measurements = len(list_measurements)
    rmse_values = [1 / measurement.calculate_rmse()**2 for measurement in list_measurements]

    rmse_matrix = np.zeros((num_measurements, num_measurements))

    np.fill_diagonal(rmse_matrix, rmse_values)

    return rmse_matrix


def assess_points_accuracy(list_of_points: List[Points],
                           list_measurements: List[Union[LinearMeasurements]]
                           ) -> Dict[str, List[float]]:
    filtered_points = create_filtered_points(list_of_points, list_measurements)
    filtered_measurements = create_filtered_measurements(filtered_points, list_measurements)

    J = create_jacobian_matrix(filtered_points, filtered_measurements)
    P = create_precision_matrix(filtered_measurements)
    K = np.linalg.inv(J.T@P@J)

    # Writing values to Point instances
    result_table = {}
    for i, point in filtered_points.items():
        if isinstance(point, (RefinedPoints, EstimatedPoints)):
            rmse_x = np.sqrt(K[i * 2, i * 2])
            rmse_y = np.sqrt(K[i * 2 + 1, i * 2 + 1])
            cov_xy = K[i * 2, i * 2 + 1]
            correlation = cov_xy / (rmse_x * rmse_y) if rmse_x * rmse_y != 0 else 0.0

            point.rmse_x = rmse_x
            point.rmse_y = rmse_y
            point.cov_xy = cov_xy

            result_table[point.name] = [point.x, point.rmse_x * 1000,
                                        point.y, point.rmse_y * 1000,
                                        correlation]

    return result_table


def main():

    result_import = assess_points_accuracy(*import_points())
    print(*result_import.items(), sep='\n')


if __name__ == '__main__':
    main()
