vmgomp.ebic.update <- function(res, n, p) {

  len <- dim(res)[1]
  sel <- 1:len
  if ( n > p )  gam <- 0 else gam <- 0.5
  for ( i in 1:len ) {
    ebic <- 2 * log(n) + 2 * gam * (p - i) / (i + 1)
    if ( 2 * (res[i + 1, 2] - res[i, 2] ) < ebic ) {
       sel[i + 1] <- 0
    }
  }
  res[sel, , drop = FALSE]
}
