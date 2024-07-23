# GeoNetAccuracy
This repository contains Python code for preliminary assessing the accuracy of geodetic (surveying) networks.

The main method for preliminary accuracy assessment is based on the concept of a covariance matrix as an inverse metric tensor in the parameter space, found using the general formula in index notation:

<p>
  $$\hat{P}_{(ij)}=J_{i}^{l}J_{j}^{k}P_{(lk)}=\frac{\partial x^{l}}{\partial \hat{x}^i}\frac{\partial x^{k}}{\partial \hat{x}^j}P_{(lk)}\quad \text{where}\quad \hat{P}_{(ik)}\hat{K}^{(kj)}=\delta^{j}_{i}$$
</p>
<p>
  where <br>
  $\quad x^i$, $\hat{x}^i$ are the components of the vector of measurement and the vector of estimated parameters, respectively, <br>
  $\quad P_{(ij)}$, $\hat{P}_{(ij)}$ are the components of the measurement accuracy matrix and the parameter accuracy matrix, respectively, <br>
  $\quad \hat{K}^{(ij)}$ are the components of the parameter covariance matrix, if $i=j$, then $\hat{K}^{(ij)}=var(\hat{x}^i)$, otherwise $\hat{K}^{(ij)}=cov(\hat{x}^i,\hat{x}^j)$.<br>
</p>

More information about tensors in statistics can be found in monograph by Prof. Peter McCullagh:
<p>McCullagh, P. (1987). Tensor methods in statistics: Monographs on statistics and applied probability (1st ed.). New York: Chapman and Hall/CRC. <a href="https://www.taylorfrancis.com/books/mono/10.1201/9781351077118/tensor-methods-statistics-mccullagh" target="_blank"><cite>doi:10.1201/9781351077118</cite></a>.</p>

More information about tensor calculus and index notation can be found in monograph by Prof. Yuri Ivanovich Dimitrienko:
<p>Dimitrienko, Y. I. (2001). Tenzornoe ischislenie [Tensor calculus]. M.: "Vysshaya shkola".</p>

More general information about the theory of measurement errors can be found in:
<p>Grodecki, J. (1997). Estimation of Variance-Covariance Components for Geodetic Observations and Implications on Deformation Trend Analysis. Ph.D. dissertation. <a href="https://gge.ext.unb.ca/Pubs/TR186.pdf" target="_blank"><cite>Engineering Technical Report No. 186, University of New Brunswick, Department of Geodesy and Geomatics Engineering, Fredericton.</cite></a></p>

<p>Gordeev, V. A. (2004). Teoriya oshibok izmerenij i uravnitelnye vychisleniya [Measurement error theory and adjustment computations] (2nd ed.). Yekaterinburg: UrSMU. ISBN 5-8019-0054-3</p>

<p>Amiri-Simkooei, A. (2007). Least-squares variance component estimation: theory and GPS applications. PhD thesis. Delft University of Technology, Delft institute of Earth Observation and Space systems (DEOS). Delf: Publications on Geodesy, 64, Netherlands Geodetic Commission. Retrieved from <a href="http://resolver.tudelft.nl/uuid:bc7f8919-1baf-4f02-b115-dc926c5ec090" target="_blank"><cite>http://resolver.tudelft.nl/uuid:bc7f8919-1baf-4f02-b115-dc926c5ec090</cite></a>.</p></p>
