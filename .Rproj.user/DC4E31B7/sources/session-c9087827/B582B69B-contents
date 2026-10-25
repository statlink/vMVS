\name{vmgomp.ebic.update}
\alias{vmgomp.ebic.update}
\title{
Update the results of the gama-Orthogonal Matching Pursuit (gamma-OMP) via eBIC
}

\description{
Update the results of the \eqn{\gamma}-OMP via eBIC.
}

\usage{
vmgomp.ebic.update(res, n, p)
}

\arguments{
\item{res}{
The matrix with the results of the \eqn{\gamma}-OMP.
}
\item{n}{
The sample size.
}
\item{p}{
The number of predictor variables.
}
}

\details{
Suppose you ran \eqn{\gamma}-OMP and now want to run it again using the eBIC as
an extra selection cterion. Instead of running it again you may simply update the
results using eBIC.
}

\value{
The updated "res"" matrix with the results of the \eqn{\gamma}-OMP.
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
\code{ \link{vm.gomp}
}
}

\examples{
x <- matrix( rnorm(500 * 50), ncol = 50)
y <- runif(500, 0, 2 * pi)
a <- vm.gomp(y, x)
b <- vm.gomp(y, x, ebic = TRUE)
}

