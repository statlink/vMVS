cv.vm.gomp <- function(y, x, rads = TRUE, alpha = 0.05, R = 499L, tol = 1e-6, maxiters = 500L,
                       ncores = 1L, nfolds = 10, folds = NULL, seed = NULL) {

  runtime <- proc.time()
  if ( !is.matrix(y) )  y <- angles_to_xy_rcpp(y, rads, ncores = 1L)
  n <- dim(y)[1]   ;   p <- dim(x)[2]
  ina <- 1:n
  if ( is.null(folds) )  folds <- Directional::makefolds(ina, nfolds = nfolds,
                                                           stratified = FALSE, seed = seed)
  nfolds <- length(folds)
  perf <- matrix(nrow = nfolds, ncol = 2)

  for ( i in 1:nfolds ) {
    xtest <- x[ folds[[ i ]], ]
    ytest <- y[ folds[[ i ]], ]
    xtrain <- x[ -folds[[ i ]], ]
    ytrain <- y[-folds[[ i ]], ]
    mod <- vMVS::vm.gomp(ytrain, xtrain, alpha = alpha, R = R, tol = tol, maxiters = maxiters, ncores = ncores)
    ntest <- dim(ytest)[1]
    res <- mod$pvalue.res[-1, 1]
    if ( length(res) == 0 ) {
      est <- Rfast::vmf.mle(y)$mu
      est <- matrix( rep(est, ntest), nrow = ntest, byrow = TRUE)
    } else {
      est <- vMVS::vm.reg(ytrain, xtrain[, res], xnew = xtest[, res], tol = tol,
                          maxiters = maxiters, ncores = ncores)$est
    }
    cs  <- Rfast::rowsums(ytest * est)
    mae <- sum( acos( pmin( pmax(cs, -1), 1 ) ) ) / ntest
    perf[i, 1] <- mae * 180 / pi
    ### eBIC now
    res <- mod$ebic.res[-1, 1]
    if ( length(res) == 0 ) {
      est <- Rfast::vmf.mle(y)$mu
      est <- matrix( rep(est, ntest), nrow = ntest, byrow = TRUE)
    } else {
      est <- vMVS::vm.reg(ytrain, xtrain[, res], xnew = xtest[, res], tol = tol,
                          maxiters = maxiters, ncores = ncores)$est
    }
    cs  <- Rfast::rowsums(ytest * est)
    mae <- sum( acos( pmin( pmax(cs, -1), 1 ) ) ) / ntest
    perf[i, 2] <- mae * 180 / pi
  }

  perf <- Rfast::colmeans(perf)
  names(perf) <- c("p-value", "eBIC")
  runtime <- proc.time() - runtime

  list(runtime = runtime, perf = perf)
}
