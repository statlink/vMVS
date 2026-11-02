cv.vm.gomp <- function(y, x, rads = TRUE, alpha = 0.05, R = 499L, tol = 1e-6, maxiters = 500L,
                       ncores = 1L, nfolds = 10, folds = NULL, seed = NULL) {

  runtime <- proc.time()
  if ( !is.matrix(y) )  y <- angles_to_xy_rcpp(y, rads, ncores = 1L)
  n <- dim(y)[1]
  ina <- 1:n
  if ( is.null(folds) )  folds <- Directional::makefolds(ina, nfolds = nfolds,
                                                         stratified = FALSE, seed = seed)
  nfolds <- length(folds)
  perf <- matrix(nrow = nfolds, ncol = 2)

  ## mean angular error (in degrees) of the predictions from a given set of selected columns
  ang.err <- function(sel, ytrain, xtrain, ytest, xtest) {
    ntest <- dim(ytest)[1]
    est <- NULL
    if ( length(sel) > 0 ) {
      est <- tryCatch( vMVS::vm.reg(ytrain, xtrain[, sel, drop = FALSE], xnew = xtest[, sel, drop = FALSE],
                                    tol = tol, maxiters = maxiters, ncores = ncores)$est,
                       error = function(e) NULL )
    }
    if ( is.null(est) ) {   ## null model (or failed fit): mean direction of the TRAINING data
      mu <- Rfast::vmf.mle(ytrain, tol = tol)$mu
      est <- matrix( mu, nrow = ntest, ncol = 2, byrow = TRUE )
    }
    cs <- Rfast::rowsums(ytest * est)
    sum( acos( pmin( pmax(cs, -1), 1 ) ) ) / ntest * 180 / pi
  }

  for ( i in 1:nfolds ) {
    te <- folds[[ i ]]
    xtest <- x[te, , drop = FALSE]    ;   ytest <- y[te, , drop = FALSE]
    xtrain <- x[-te, , drop = FALSE]  ;   ytrain <- y[-te, , drop = FALSE]
    mod <- vMVS::vm.gomp(ytrain, xtrain, alpha = alpha, R = R, tol = tol,
                         maxiters = maxiters, ncores = ncores)
    sel.p <- mod$pvalue.res[-1, 1]
    sel.e <- if ( is.null(mod$ebic.res) )  numeric(0)  else  mod$ebic.res[-1, 1]
    perf[i, 1] <- ang.err(sel.p, ytrain, xtrain, ytest, xtest)
    ## eBIC selects a prefix of the p-value path; skip the refit when it is the same set
    if ( length(sel.e) == length(sel.p) ) {
      perf[i, 2] <- perf[i, 1]
    } else  perf[i, 2] <- ang.err(sel.e, ytrain, xtrain, ytest, xtest)
  }

  perf <- Rfast::colmeans(perf)
  names(perf) <- c("p-value", "eBIC")
  runtime <- proc.time() - runtime

  list(runtime = runtime, perf = perf)
}
