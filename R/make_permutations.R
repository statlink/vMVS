make_permutations <- function(n, R) {
  perm_idx <- matrix(NA_integer_, nrow = n, ncol = R)

  for (b in seq_len(R)) {
    perm_idx[, b] <- sample.int(n, n)
  }

  perm_idx
}
