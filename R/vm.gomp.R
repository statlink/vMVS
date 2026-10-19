# vm.gomp <- function(y, x, rads = TRUE, xstand = TRUE, alpha = 0.05, R = 499) {
#
#   runtime <- proc.time()
#   if ( !is.matrix(y) ) {
#     if ( !rads )   y <- y * pi/180
#     y <- cbind( cos(y), sin(y) )
#   }
#   if ( xstand )  x <- Rfast::standardise(x)
#   dm <- dim(x)   ;   n <- dm[1]   ;   p <- dm[2]
#
#   ind <- 1:p
#   lik <- Rfast::vmf.mle(y, tol = 1e-6)$loglik
#   ela <- Rfast::eachcol.apply(x, y[, 1])^2 + Rfast::eachcol.apply(x, y[, 2])^2
#   sel <- which.max(ela)
#   sela <- NULL
#   rho <- c( 0, .perm.reg0(y, x[, sel], R = R) )
#
#   i <- 2
#   while ( rho[i] < alpha  &  length(sela) < p - 1  &  i < n ) {
#     i <- i + 1
#     sela <- c(sela, sel)
#     ind[sela] <- 0
#     mod <- vm.reg(y, x[, sela], xnew = x[, sela])
#     lik[i - 1] <- mod$loglik
#     res <- y - mod$est * mod$A1
#     r <- numeric(p)
#     ii <- ind[ind > 0]
#     r[ind] <- Rfast::eachcol.apply(x, res[, 1], indices = ii)^2 +
#               Rfast::eachcol.apply(x, res[, 2], indices = ii)^2
#     sel <- which.max(r)
#     rho[i] <- .perm.reg1(y, x = x[, sel], xsel = x[, sela], R = R)
#   }
#
#   mod <- vm.reg(y, x[, sela])
#   runtime <- proc.time() - runtime
#   len <- length(sela)
#   res <- cbind( c(0, sela), lik[1:(len + 1)], c(NA, rho[seq_len(len) + 1]) )
#   colnames(res) <- c("Selected", "log-likelihood", "p-value")
#   list(runtime = runtime, res = res, mod = mod)
# }

vm.gomp <- function(y, x, rads = TRUE, xstand = TRUE, alpha = 0.05, R = 499L,
                    tol = 1e-6, maxiters = 500L, ncores = 1L) {

  runtime <- proc.time()
  if (!is.matrix(y)) {
    # ncores>1 has gains only for about n>2500
    y <- angles_to_xy_rcpp(y, rads, ncores = 1L)
  }

  if (xstand) x <- Rfast::standardise(x)

  dm <- dim(x)
  n <- dm[1]
  p <- dm[2]

  ind <- 1:p

  lik <- Rfast::vmf.mle(y, tol = tol)$loglik

  ela <- Rfast::eachcol.apply(x, y[, 1])^2 +
    Rfast::eachcol.apply(x, y[, 2])^2

  sel <- which.max(ela)
  sela <- NULL

  rho <- c(
    0,
    perm.reg0(
      y, x[, sel, drop = FALSE],
      R = R,
      tol = tol,
      maxiters = maxiters,
      ncores = ncores
    )
  )

  i <- 2

  while (rho[i] < alpha & length(sela) < p - 1 & i < n) {

    i <- i + 1

    sela <- c(sela, sel)
    ind[sela] <- 0

    mod <- vm.reg(
      y, x[, sela, drop = FALSE],
      xnew = x[, sela, drop = FALSE],
      tol = tol,
      maxiters = maxiters,
      ncores = ncores
    )

    lik[i - 1] <- mod$loglik

    res <- y - mod$est * mod$A1

    r <- numeric(p)
    ii <- ind[ind > 0]

    r[ind] <- Rfast::eachcol.apply(
      x, res[, 1], indices = ii
    )^2 +
      Rfast::eachcol.apply(
        x, res[, 2], indices = ii
      )^2

    sel <- which.max(r)

    rho[i] <- perm.reg1(
      y,
      x = x[, sel, drop = FALSE],
      xsel = x[, sela, drop = FALSE],
      R = R,
      tol = tol,
      maxiters = maxiters,
      ncores = ncores
    )
  }

  mod <- vm.reg(
    y, x[, sela, drop = FALSE],
    tol = tol,
    maxiters = maxiters,
    ncores = ncores
  )

  runtime <- proc.time() - runtime

  len <- length(sela)

  res <- cbind(
    c(0, sela),
    lik[1:(len + 1)],
    c(NA, rho[seq_len(len) + 1])
  )

  colnames(res) <- c("Selected", "log-likelihood", "p-value")

  list(runtime = runtime, res = res, mod = mod)
}




