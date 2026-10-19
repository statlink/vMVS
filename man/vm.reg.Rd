\name{vm.reg}
\alias{vm.reg}
\title{
von Mises regression model
}

\description{
von Mises regression model.
}

\usage{
vm.reg(y, x, rads = TRUE, xnew = NULL, tol = 1e-6, maxiters = 500L, ncores = 1L)
}

\arguments{
\item{y}{
A vector with the circular data expressed in radians, or angles. It can also
be a matrix with Euclidean representation of the angles.
}
\item{x}{
The independent variable(s). Can be Euclidean or categorical (factor variables).
}
\item{rads}{
If the response is expressed in angles set this to FALSE.
}
\item{xnew}{
The new values of some independent variable(s) whose circular values you want
to predict. The can be Euclidean or categorical. If you have no new x values,
leave it NULL (default).
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
}

\details{
The functions performs regression using circular distributions.
}

\value{
A list including:
\item{be}{
A matrix with the estimated regression coefficients
}
\item{loglik}{
The value of the maximised log-likelihood of the chosen distribution.
}
\item{ki}{
The concentration parameters for each observation.
}
\item{A1}{
The \eqn{A_1} function for each observation.
}
\item{est}{
If xnew is not NULL, this contains the estimated response values in their Euclidean
representation.
}
}

\references{
P. J. Paine, S. P. Preston, M. Tsagris and Andrew T. A. Wood (2020).
Spherical regression models with general covariates and anisotropic errors.
Statistics and Computing, 30(1): 153--165.

Tsagris M., Papadovasilakis Z., Lakiotaki K. and Tsamardinos I. (2022).
The \eqn{\gamma}-OMP algorithm for feature selection with application to gene
expression data. IEEE/ACM Transactions on Computational Biology and
Bioinformatics, 19(2): 1214--1224.
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}
%\note{
%%  ~~further notes~~
%}
\seealso{
\code{\link{vm.gomp}
}
}

\examples{
x <- matrix( rnorm(400), ncol = 4)
y <- runif(100, 0, pi)
mod <- vm.reg(y, x)
}
