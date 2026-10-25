\name{vm.gomp}
\alias{vm.gomp}
\title{
The gama-Orthogonal Matching Pursuit (gamma-OMP) algorithm for the von Mises
regression
}

\description{
The \eqn{\gamma}-Orthogonal Matching Pursuit (\eqn{\gamma}-OMP) algorithm for
the von Mises regression.
}

\usage{
vm.gomp(y, x, rads = TRUE, xstand = TRUE, alpha = 0.05, R = 499L, ebic = FALSE,
model = FALSE, tol = 1e-6, maxiters = 500L, ncores = 1L)
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
\item{xstand}{
If this is TRUE the independent variables are standardised.
}
\item{alpha}{
The significance level.
}
\item{R}{
How many permutations to perform for the hypothesis testing of each candidate
predictor variable
}
\item{ebic}{
If you set this equal to TRUE, then it uses the exteded BIC to retain the candidate variable,
even if it its p-value is less than the selected threshold.
}
\item{model}{
Do you want to return the model with the final selected variables?
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
The \eqn{\gamma}-OMP adopted to the von Mises regression.
}

\value{
A list including:
\item{runtime}{
The runtime of the algorithm
}
\item{res}{
A matrix with three columns. The selected variable(s), the log-likelihood
with each selected variable and the p-value of each selected variable.
}
\item{mod}{
The output of the function \code{ \link{vm.reg} } where the model is fitted with
the selected variables, if the argument "model" was set equal to TRUE.
}
}

\references{
Tsagris M., Papadovasilakis Z., Lakiotaki K. and Tsamardinos I. (2022).
The \eqn{\gamma}-OMP algorithm for feature selection with application to gene
expression data. IEEE/ACM Transactions on Computational Biology and
Bioinformatics, 19(2): 1214--1224.

Paine P.J., Preston, S.P. Tsagris M. and Wood Andrew T.A. (2020).
Spherical regression models with general covariates and anisotropic errors.
Statistics and Computing, 30(1): 153--165.

Chen J. and Chen Z. (2012). Extended BIC for small-n-large-P sparse GLM.
Statistica Sinica, 22(2): 555--574.
}

\author{
Michail Tsagris and Christos Adam.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr} and
Christos Adam \email{econp266@econ.soc.uoc.gr}.
}
%\note{
%%  ~~further notes~~
%}

\seealso{
\code{ \link{vmgomp.ebic.update}, \link{vm.reg}
}
}

\examples{
x <- matrix( rnorm(500 * 50), ncol = 50)
y <- runif(500, 0, 2 * pi)
a <- vm.gomp(y, x)
b <- vm.gomp(y, x, ebic = TRUE)
}

