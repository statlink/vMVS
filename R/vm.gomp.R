vm.gomp <- function(y, x, rads = TRUE, alpha = 0.05, R = 499L, tol = 1e-6,
                    maxiters = 500L, ncores = 1L, ctol = 1e-6) {

  runtime <- proc.time()
  if ( !is.matrix(y) )  y <- angles_to_xy_rcpp(y, rads, ncores = 1L)

  dm <- dim(x)
  n <- dm[1]   ;   p <- dm[2]
  ind <- 1:p

  const <- which( vx < .Machine$double.eps )
  ssx <- n - 1
  if ( length(const) > 0 ) {
    x[, const] <- 0   ;   ind[const] <- 0
  }

  pick <- function(res, ii, cols = NULL) {
    if ( length(ii) == 0 )  return( list(sel = NA, drop = NULL) )
    r <- Rfast::eachcol.apply(x, res[, 1], indices = ii)^2 +
         Rfast::eachcol.apply(x, res[, 2], indices = ii)^2
    o <- order(r, decreasing = TRUE)
    cand <- ii[o][ r[o] > 0 ]
    if ( is.null(cols) )  return( list(sel = cand[1], drop = NULL) )
    qrs <- qr( cbind(1, x[, cols, drop = FALSE]) )
    drop <- NULL
    for ( k in cand ) {
      e <- qr.resid( qrs, x[, k] )
      if ( sum(e^2) / ssx > ctol )  return( list(sel = k, drop = drop) )
      drop <- c(drop, k)
    }
    list(sel = NA, drop = drop)
  }

  ## von Mises regression on the given columns; NULL if it fails or is non-finite
  fitvm <- function(cols) {
    xs <- x[, cols, drop = FALSE]
    m <- tryCatch( vMVS::vm.reg(y, xs, xnew = xs, tol = tol, maxiters = maxiters, ncores = ncores),
                   error = function(e) NULL )
    if ( is.null(m) || !is.finite(m$loglik) || any( !is.finite(m$est) ) )  NULL  else  m
  }

  ## p-value of a permutation test; 1 (i.e. stop) if it fails or is non-finite
  ptest <- function(expr) {
    v <- tryCatch( expr, error = function(e) NA )
    if ( length(v) != 1 || !is.finite(v) )  1  else  v
  }

  lik <- Rfast::vmf.mle(y, tol = tol)$loglik
  yc <- Rfast::eachrow( y, Rfast::colmeans(y), oper = "-" )
  sel <- pick( yc, ind[ind > 0] )$sel
  sela <- NULL
  if ( is.na(sel) ) {
    rho <- c(0, 1)
  } else  rho <- c( 0, ptest( vMVS::perm.reg0( y, x[, sel, drop = FALSE], R = R, tol = tol,
                                               maxiters = maxiters, ncores = ncores ) ) )

  i <- 2
  while ( rho[i] < alpha  &  length(sela) < p - 1  &  i < n ) {
    i <- i + 1
    sela <- c(sela, sel)
    ind[sela] <- 0

    mod <- fitvm(sela)
    if ( is.null(mod) ) {
      sela <- sela[-length(sela)]
      rho[i] <- 1
      break
    }
    lik[i - 1] <- mod$loglik
    res <- y - mod$est * mod$A1

    pk <- pick( res, ind[ind > 0], sela )
    ind[pk$drop] <- 0   ## collinear with the selected set now, so also later (the span only grows)
    sel <- pk$sel

    if ( is.na(sel) ) {
      rho[i] <- 1
    } else  rho[i] <- ptest( vMVS::perm.reg1( y, x = x[, sel, drop = FALSE], xsel = x[, sela, drop = FALSE],
                                              R = R, tol = tol, maxiters = maxiters, ncores = ncores ) )
  }

  len <- length(sela)
  pvalue.res <- cbind( c(0, sela), lik[1:(len + 1)], c(NA, rho[seq_len(len) + 1]) )
  colnames(pvalue.res) <- c("Selected", "log-likelihood", "p-value")

  ## eBIC (Chen and Chen, 2012) along the forward path; each predictor adds 2 parameters
  ebic.res <- NULL
  if ( len > 0 ) {
    k <- 0:len
    if ( n > p ) {
      gam <- 0
    } else  gam <- min( 1, 1 - log(n) / (2 * log(p)) + 0.05 )
    ebic <- -2 * pvalue.res[, 2] + 2 * k * log(n) + 2 * gam * lchoose(p, k)
    khat <- which.min(ebic)
    ebic.res <- cbind( pvalue.res[1:khat, , drop = FALSE], eBIC = ebic[1:khat] )
  }

  runtime <- proc.time() - runtime
  list(runtime = runtime, pvalue.res = pvalue.res, ebic.res = ebic.res)
}





# vm.gomp <- function(y, x, rads = TRUE, xstand = TRUE, alpha = 0.05, R = 499L,
#                     tol = 1e-6, maxiters = 500L, ncores = 1L) {
#
#   runtime <- proc.time()
#   if ( !is.matrix(y) )  y <- angles_to_xy_rcpp(y, rads, ncores = 1L)
#   if ( xstand ) x <- Rfast::standardise(x)
#
#   dm <- dim(x)
#   n <- dm[1]   ;   p <- dm[2]
#   ind <- 1:p
#
#   lik <- Rfast::vmf.mle(y, tol = tol)$loglik
#   ela <- Rfast::eachcol.apply(x, y[, 1])^2 + Rfast::eachcol.apply(x, y[, 2])^2
#   sel <- which.max(ela)
#   sela <- NULL
#   rho <- c( 0, vMVS::perm.reg0( y, x[, sel, drop = FALSE], R = R,
#             tol = tol, maxiters = maxiters, ncores = ncores) )
#   i <- 2
#   #x2 <- x^2        # once, outside the loop
#
#   while ( rho[i] < alpha  &  length(sela) < p - 1  &  i < n ) {
#     i <- i + 1
#     sela <- c(sela, sel)
#     ind[sela] <- 0
#     mod <- vMVS::vm.reg(y, x[, sela, drop = FALSE], xnew = x[, sela, drop = FALSE],
#                   tol = tol, maxiters = maxiters, ncores = ncores)
#     lik[i - 1] <- mod$loglik
#     res <- y - mod$est * mod$A1
#   	#ii <- ind[ind > 0]
#     #U  <- cbind( Rfast::eachcol.apply(x, res[, 1], indices = ii), Rfast::eachcol.apply(x, res[, 2], indices = ii) )
#     #M  <- crossprod( x2[, ii, drop = FALSE], cbind(mod$d11, mod$d22, mod$d12) )
#     #r[ii] <- ( U[, 1]^2 * M[, 2] - 2 * U[, 1] * U[, 2] * M[, 3] + U[, 2]^2 * M[, 1] ) / ( M[, 1] * M[, 2] - M[, 3]^2 )
#   	r <- numeric(p)
#     ii <- ind[ind > 0]
#     r[ind] <- Rfast::eachcol.apply(x, res[, 1], indices = ii)^2 + Rfast::eachcol.apply(x, res[, 2], indices = ii)^2
#     sel <- which.max(r)
#     rho[i] <- vMVS::perm.reg1(y, x = x[, sel, drop = FALSE], xsel = x[, sela, drop = FALSE],
#                         R = R, tol = tol, maxiters = maxiters, ncores = ncores)
#   }
#
#   len <- length(sela)
#   pvalue.res <- cbind( c(0, sela), lik[1:(len + 1)], c(NA, rho[seq_len(len) + 1]) )
#   colnames(pvalue.res) <- c("Selected", "log-likelihood", "p-value")
#
#   res <- pvalue.res   ;   ebic.res <- NULL
#   if ( dim(res)[1] > 1 ) {
#     sel <- 1:c(len + 1)
#     if ( n > p )  gam <- 0  else  gam <- 0.5
#     for ( i in 1:len ) {
#       ebic <- 2 * log(n) + 2 * gam * (p - i) / (i + 1)
#       if ( 2 * (res[i + 1, 2] - res[i, 2] ) < ebic ) {
#         sel[i + 1] <- 0
#       }
#     }
#     ebic.res <- res[sel, , drop = FALSE]
#    }
#
#   runtime <- proc.time() - runtime
#   list(runtime = runtime, pvalue.res = pvalue.res, ebic.res = ebic.res)
# }


