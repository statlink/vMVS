# perm.reg1 <- function(y, x, xsel, tol = 1e-6, maxiters = 500, R = 999) {
#   n <- dim(y)[1]
#   xf <- cbind(xsel, x)
#
#   mod0 <- vm.reg(y = y, x = xsel, xnew = xsel, tol = tol, maxiters = maxiters)
#   u <- mod0$est
#   stat <- vm.reg(y = y, x = xf, tol = tol, maxiters = maxiters)$loglik - mod0$loglik
#
#   e <- cbind( y[, 1] * u[, 1] + y[, 2] * u[, 2], y[, 2] * u[, 1] - y[, 1] * u[, 2] )
#
#   bstat <- numeric(R)
#   for ( i in 1:R ) {
#     eb <- e[sample.int(n, n), ]
#     yb <- cbind( eb[, 1] * u[, 1] - eb[, 2] * u[, 2], eb[, 1] * u[, 2] + eb[, 2] * u[, 1] )
#     bstat[i] <- vm.reg(y = yb, x = xf, tol = tol, maxiters = maxiters)$loglik -
#       vm.reg(y = yb, x = xsel, tol = tol, maxiters = maxiters)$loglik
#   }
#   ( sum(bstat >= stat) + 1 ) / (R + 1)
# }

perm.reg1 <- function(y, x, xsel, R = 999L, tol = 1e-6, maxiters = 500L, ncores = 1L) {

  stopifnot(
    length(ncores) == 1L,
    !is.na(ncores),
    ncores >= 1,
    ncores == as.integer(ncores),
    length(R) == 1L,
    !is.na(R),
    R >= 1,
    R == as.integer(R)
  )

  ncores <- as.integer(ncores)
  R <- as.integer(R)

  y <- as.matrix(y)
  x <- as.matrix(x)
  xsel <- as.matrix(xsel)

  n <- nrow(y)

  stopifnot(
    ncol(y) == 2L,
    nrow(x) == n,
    nrow(xsel) == n
  )

  # Full-model predictors, excluding the intercept,
  # which is added in C++.
  xf <- cbind(xsel, x)

  # Fit the reduced model.
  mod0 <- vm.reg(
    y, xsel, xnew = xsel,
    tol = tol,
    maxiters = maxiters,
    ncores = ncores
  )

  u <- mod0$est

  # Observed likelihood-ratio statistic.
  ll_full <- vm_reg_loglik_export(
    y, xf,
    tol = tol,
    maxiters = maxiters
  )

  stat <- ll_full - mod0$loglik

  # Generate the permutations once, outside C++ workers.
  perm_idx <- make_permutations(n, R)

  if (ncores == 1L) {
    return(perm_reg1_serial_fast_cpp(
      y, xsel, xf, u, stat,
      perm_idx = perm_idx,
      tol = tol,
      maxiters = maxiters
    ))
  }

  perm_reg1_parallel_fast_cpp(
    y, xsel, xf, u, stat,
    perm_idx = perm_idx,
    tol = tol,
    maxiters = maxiters,
    ncores = ncores
  )
}
