# perm.reg0 <- function(y, x, tol = 1e-6, maxiters = 500, R = 999) {
#
#   n <- dim(y)[1]
#   lik0 <- Rfast::vmf.mle(y, tol = tol)$loglik
#   stat <- vm.reg(y = y, x = x, tol = tol, maxiters = maxiters)$loglik - lik0
#
#   bstat <- numeric(R)
#   for ( i in 1:R ) {
#     yb <- y[sample.int(n, n), ]
#     bstat[i] <- vm.reg(y = yb, x = x, tol = tol, maxiters = maxiters)$loglik - lik0
#   }
#   ( sum(bstat >= stat) + 1 ) / (R + 1)
# }

perm.reg0 <- function(y, x, R = 999L, tol = 1e-6, maxiters = 500L, ncores = 1L) {
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

  stopifnot(
    ncol(y) == 2L,
    nrow(x) == nrow(y)
  )

  # Observed likelihood-ratio statistic.
  lik0 <- Rfast::vmf.mle(y, tol = tol)$loglik

  lik1 <- vm_reg_loglik_export(
    y, x, tol = tol, maxiters = maxiters
  )

  stat <- lik1 - lik0

  # Generate permutations once on R's main thread.
  perm_idx <- make_permutations(nrow(y), R)

  if (ncores == 1L) {
    return(perm_reg0_serial_idx_cpp(
      y, x,
      lik0 = lik0,
      stat = stat,
      perm_idx = perm_idx,
      tol = tol,
      maxiters = maxiters
    ))
  }

  perm_reg0_parallel_cpp(
    y, x,
    lik0 = lik0,
    stat = stat,
    perm_idx = perm_idx,
    tol = tol,
    maxiters = maxiters,
    ncores = ncores
  )
}
