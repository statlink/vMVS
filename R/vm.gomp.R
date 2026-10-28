vm.gomp <- function(y, x, rads = TRUE, xstand = TRUE, alpha = 0.05, R = 499L,
                    tol = 1e-6, maxiters = 500L, ncores = 1L) {

  runtime <- proc.time()
  if ( !is.matrix(y) )  y <- angles_to_xy_rcpp(y, rads, ncores = 1L)
  if ( xstand ) x <- Rfast::standardise(x)

  dm <- dim(x)
  n <- dm[1]   ;   p <- dm[2]
  ind <- 1:p

  lik <- Rfast::vmf.mle(y, tol = tol)$loglik
  ela <- Rfast::eachcol.apply(x, y[, 1])^2 + Rfast::eachcol.apply(x, y[, 2])^2
  sel <- which.max(ela)
  sela <- NULL
  rho <- c( 0, vMVS::perm.reg0( y, x[, sel, drop = FALSE], R = R,
            tol = tol, maxiters = maxiters, ncores = ncores) )
  i <- 2
  #x2 <- x^2        # once, outside the loop

  while ( rho[i] < alpha  &  length(sela) < p - 1  &  i < n ) {
    i <- i + 1
    sela <- c(sela, sel)
    ind[sela] <- 0
    mod <- vMVS::vm.reg(y, x[, sela, drop = FALSE], xnew = x[, sela, drop = FALSE],
                  tol = tol, maxiters = maxiters, ncores = ncores)
    lik[i - 1] <- mod$loglik
    res <- y - mod$est * mod$A1
  	#ii <- ind[ind > 0]
    #U  <- cbind( Rfast::eachcol.apply(x, res[, 1], indices = ii), Rfast::eachcol.apply(x, res[, 2], indices = ii) )
    #M  <- crossprod( x2[, ii, drop = FALSE], cbind(mod$d11, mod$d22, mod$d12) )
    #r[ii] <- ( U[, 1]^2 * M[, 2] - 2 * U[, 1] * U[, 2] * M[, 3] + U[, 2]^2 * M[, 1] ) / ( M[, 1] * M[, 2] - M[, 3]^2 )
  	r <- numeric(p)
    ii <- ind[ind > 0]
    r[ind] <- Rfast::eachcol.apply(x, res[, 1], indices = ii)^2 + Rfast::eachcol.apply(x, res[, 2], indices = ii)^2
    sel <- which.max(r)
    rho[i] <- vMVS::perm.reg1(y, x = x[, sel, drop = FALSE], xsel = x[, sela, drop = FALSE],
                        R = R, tol = tol, maxiters = maxiters, ncores = ncores)
  }

  len <- length(sela)
  pvalue.res <- cbind( c(0, sela), lik[1:(len + 1)], c(NA, rho[seq_len(len) + 1]) )
  colnames(pvalue.res) <- c("Selected", "log-likelihood", "p-value")

  res <- pvalue.res   ;   ebic.res <- NULL
  if ( dim(res)[1] > 1 ) {
    sel <- 1:c(len + 1)
    if ( n > p )  gam <- 0  else  gam <- 0.5
    for ( i in 1:len ) {
      ebic <- 2 * log(n) + 2 * gam * (p - i) / (i + 1)
      if ( 2 * (res[i + 1, 2] - res[i, 2] ) < ebic ) {
        sel[i + 1] <- 0
      }
    }
    ebic.res <- res[sel, , drop = FALSE]
   }

  runtime <- proc.time() - runtime
  list(runtime = runtime, pvalue.res = pvalue.res, ebic.res = ebic.res)
}




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

