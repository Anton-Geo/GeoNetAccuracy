import math
import numpy as np


class Points:
    def __init__(self, type_name: str, name: str, x: float, y: float,
                 rmse_x: float = 1000.0, rmse_y: float = 1000.0, corr_xy: float = 0.0):
        self._type_name = type_name
        self._name = name
        self._x = x
        self._y = y
        self._rmse_x = rmse_x
        self._rmse_y = rmse_y
        self._corr_xy = corr_xy
        self._rmse_major = None
        self._rmse_minor = None
        self._ellipse_angle = None

    def calc_ellipse_parameters(self):
        cov_matrix = np.array([[self._rmse_x**2, self._corr_xy * self._rmse_x * self._rmse_y],
                               [self._corr_xy * self._rmse_x * self._rmse_y, self._rmse_y**2]])
        tr_c_m, det_c_m = np.trace(cov_matrix), np.linalg.det(cov_matrix)
        rmse_major = ((tr_c_m + (tr_c_m**2 - 4 * det_c_m)**0.5) / 2)**0.5
        rmse_minor = ((tr_c_m - (tr_c_m**2 - 4 * det_c_m)**0.5) / 2)**0.5
        ellipse_angle = 0.5 * math.atan2(2 * self._corr_xy * self._rmse_x * self._rmse_y,
                                         self._rmse_y**2 - self._rmse_x**2)
        # math.atan2 already takes into account the sign, so no additional condition is required
        # if self._rmse_y > self._rmse_x:
        #     ellipse_angle += math.pi / 2
        return rmse_major, rmse_minor, ellipse_angle

    def update_ellipse_parameters(self):
        self._rmse_major, self._rmse_minor, self._ellipse_angle = self.calc_ellipse_parameters()

    @property
    def type_name(self) -> str:
        return self._type_name

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
    def corr_xy(self) -> float:
        return self._corr_xy

    @property
    def rmse_major(self) -> float:
        return self._rmse_major

    @property
    def rmse_minor(self) -> float:
        return self._rmse_minor

    @property
    def ellipse_angle(self) -> float:
        return self._ellipse_angle

    @type_name.setter
    def type_name(self, new_type_name: str):
        self._type_name = new_type_name

    @name.setter
    def name(self, new_name: str):
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
        self._rmse_major, self._rmse_minor, self._ellipse_angle = self.calc_ellipse_parameters()

    @rmse_y.setter
    def rmse_y(self, new_rmse_y: float):
        self._rmse_y = new_rmse_y
        self._rmse_major, self._rmse_minor, self._ellipse_angle = self.calc_ellipse_parameters()

    @corr_xy.setter
    def corr_xy(self, new_corr_xy: float):
        self._corr_xy = new_corr_xy
        self._rmse_major, self._rmse_minor, self._ellipse_angle = self.calc_ellipse_parameters()

    @rmse_major.setter
    def rmse_major(self, new_rmse_major: float):
        self._rmse_major = new_rmse_major

    @rmse_minor.setter
    def rmse_minor(self, new_rmse_minor: float):
        self._rmse_minor = new_rmse_minor

    @ellipse_angle.setter
    def ellipse_angle(self, new_ellipse_angle: float):
        self._ellipse_angle = new_ellipse_angle


class BasePoints(Points):
    def __init__(self, name: str, x: float, y: float):
        super().__init__(type_name='base_point', name=name,
                         x=x, y=y,
                         rmse_x=0.0, rmse_y=0.0, corr_xy=0.0)


class RefinedPoints(Points):
    def __init__(self, name: str, x: float, y: float,
                 rmse_x: float, rmse_y: float, corr_xy: float):
        super().__init__(type_name='refined_point', name=name,
                         x=x, y=y,
                         rmse_x=rmse_x, rmse_y=rmse_y, corr_xy=corr_xy)


class EstimatedPoints(Points):
    def __init__(self, name: str, x: float, y: float):
        super().__init__(type_name='estimated_point', name=name,
                         x=x, y=y)
