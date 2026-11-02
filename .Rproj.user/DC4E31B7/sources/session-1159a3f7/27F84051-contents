\name{cv.vm.gomp}
\alias{cv.vm.gomp}
\title{
Cross-validation for \eqn{gamma}-OMP algorithm for the von Mises
regression
}

\description{
Cross-validation for \eqn{gamma}-OMP algorithm for the von Mises
regression.
}

\usage{
cv.vm.gomp(y, x, rads = TRUE, alpha = 0.05, R = 499L, tol = 1e-6, maxiters = 500L,
ncores = 1L, nfolds = 10, folds = NULL, seed = NULL)
}

\arguments{
\item{y}{
A vector with the circular data expressed in radians, or angles. It can also
be a matrix with Euclidean representation of the angles.
}
\item{x}{
A matrix with continuous data, where the rows denote the observations and the
columns are the variables.
}
\item{rads}{
If the response is expressed in angles set this to FALSE.
}
\item{alpha}{
The significance level.
}
\item{R}{
How many permutations to perform for the hypothesis testing of each candidate
predictor variable
}
\item{tol}{
The tolerance value to terminate the Newton-Raphson algorithm.
}
\item{maxiters}{
The maximum number of iterations allowed in the Newton-Raphson algorithm.
}
\item{ncores}{
The number of cores to use. If greater than 1, parallel computing will take
place.
}
\item{nfolds}{
The number of folds to create.
}
\item{folds}{
If you aleady have the list with the folds pass it here.
}
\item{seed}{
If you want the folds to be the same type a seed number here.
}
}

\value{
A list including
\item{runtime}{
The runtime of the algorithm.
}
\item{perf}{
A numeric vector with two elements, the estimated performance,
when the significance testing and when the eBIC is used. The performance is
given by \eqn{\frac{180}{\pi}\frac{1}{n_{test}}\sum_{i=1}^{n_{test}}\cos^{-1}\left(y_i^\top\hat{y}_i\right)}.
}
}

\references{
Tsagris M., Papadovasilakis Z., Lakiotaki K. and Tsamardinos I. (2018). Efficient feature selection on gene expression data: Which algorithm to use? BioRxiv.

Tsagris M., Papadovasilakis Z., Lakiotaki K. and Tsamardinos I. (2022).
The \eqn{\gamma}-OMP algorithm for feature selection with application to gene expression data.
IEEE/ACM Transactions on Computational Biology and Bioinformatics, 19(2): 1214--1224.

Paine P.J., Preston, S.P. Tsagris M. and Wood Andrew T.A. (2020).
Spherical regression models with general covariates and anisotropic errors.
Statistics and Computing, 30(1): 153--165.

Chen J. and Chen Z. (2012). Extended BIC for small-n-large-P sparse GLM.
Statistica Sinica, 22(2): 555--574.}

\author{
Michail Tsagris

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}
%\note{
%%  ~~further notes~~
%}

\seealso{
\code{ \link{vm.gomp} }
}

\examples{
x <- matrix( rnorm(200 * 20), ncol = 20)
y <- runif(200, 0, 2 * pi)
a <- cv.vm.gomp(y, x)
}

