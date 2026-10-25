\name{MLE of the zero adjusted Dirichlet distribution}
\alias{zad.mle}
\title{
MLE of the zero adjusted Dirichlet distribution
}

\description{
MLE of the zero adjusted Dirichlet distribution.
}

\usage{
zad.mle(y)
}

\arguments{
\item{y}{
A matrix with the compositional data.
}
}

\details{
A zero adjusted Dirichlet distribution is being fitted and its parameters are estimated.
}

\value{
A list including:
\item{loglik}{
The value of the log-likelihood.
}
\item{phi}{
The precision parameter. If covariates are linked with it (function "diri.reg2"), this will
be a vector.
}
\item{mu}{
The mean vector of the distribution.
}
\item{patterns}{
A matrix with the patterns of zeros, where the value of 0 indicates the presence of a zero,
and the last column contains the percentage of occurrence each pattern. This is useful
for the random values simulation.
}
\item{runtime}{
The time required by the model..
}
}

\references{
Tsagris M. and Stewart C. (2018). A Dirichlet regression model for compositional data with zeros.
Lobachevskii Journal of Mathematics, 39(3): 398--412.

Preprint available from https://arxiv.org/pdf/1410.5011.pdf
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

%\note{
%%  ~~further notes~~
%}

\seealso{
\code{\link{zadr}
}
}

\examples{
y <- as.matrix(iris[, 1:3])
y <- y / rowSums(y)
y[sample(1:450, 15) ] <- 0
y <- y / rowSums(y)
mod <- zad.mle(y)
}

