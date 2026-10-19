#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <gsl/gsl_sf_bessel.h>
#include <cmath>
#include <vector>

// Parallelization strategy: independent permutations are parallelized with
// RcppParallel. Likelihood internals remain serial within each permutation
// worker to avoid nested parallelism and CPU oversubscription. GSL scaled
// Bessel calls are thread-safe for independent scalar evaluations.

using namespace Rcpp;
using namespace arma;

namespace {

inline void row_norm2(const arma::mat& z, arma::vec& out) {
  const arma::uword n = z.n_rows;
  out.set_size(n);
  const double* z0 = z.colptr(0);
  const double* z1 = z.colptr(1);
  for (arma::uword i = 0; i < n; ++i)
    out[i] = z0[i] * z0[i] + z1[i] * z1[i];
}

// GSL scaled modified Bessel functions.
// For kappa >= 0:
//   I0_scaled(kappa) = exp(-kappa) * I0(kappa)
//   I1_scaled(kappa) = exp(-kappa) * I1(kappa)
// These are direct GSL scaled evaluations, avoiding overflow from first
// computing the unscaled Bessel functions. GSL calls below are used in both
// serial code and RcppParallel workers; inputs are nonnegative norms.

inline void compute_I0(const arma::vec& ki, arma::vec& I0) {
  const arma::uword n = ki.n_elem;
  I0.set_size(n);
  for (arma::uword i = 0; i < n; ++i)
    I0[i] = gsl_sf_bessel_I0_scaled(ki[i]);
}

inline void compute_A1(const arma::vec& ki,
                       const arma::vec& I0,
                       arma::vec& A1) {
  const arma::uword n = ki.n_elem;
  A1.set_size(n);
  for (arma::uword i = 0; i < n; ++i)
    A1[i] = gsl_sf_bessel_I1_scaled(ki[i]) / I0[i];
}

// Keep separate names for the worker path to preserve the existing call
// structure. These call only GSL functions and do not call the R API.
inline void compute_I0_parallel_safe(const arma::vec& ki, arma::vec& I0) {
  compute_I0(ki, I0);
}

inline void compute_A1_parallel_safe(const arma::vec& ki,
                                     const arma::vec& I0,
                                     arma::vec& A1) {
  compute_A1(ki, I0, A1);
}

// Workers below operate only on C++/Armadillo memory and GSL. They do not
// call the R API, so they can safely be used by RcppParallel.
struct RowNorm2Worker : public RcppParallel::Worker {
  const double* z0;
  const double* z1;
  double* out;
  RowNorm2Worker(const arma::mat& z, arma::vec& out_)
    : z0(z.colptr(0)), z1(z.colptr(1)), out(out_.memptr()) {}
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i)
      out[i] = z0[i] * z0[i] + z1[i] * z1[i];
  }
};

struct BesselI0Worker : public RcppParallel::Worker {
  const double* ki;
  double* out;
  BesselI0Worker(const arma::vec& ki_, arma::vec& out_)
    : ki(ki_.memptr()), out(out_.memptr()) {}
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i)
      out[i] = gsl_sf_bessel_I0_scaled(ki[i]);
  }
};

struct BesselA1Worker : public RcppParallel::Worker {
  const double* ki;
  const double* I0;
  double* out;
  BesselA1Worker(const arma::vec& ki_, const arma::vec& I0_,
                 arma::vec& out_)
    : ki(ki_.memptr()), I0(I0_.memptr()), out(out_.memptr()) {}
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i)
      out[i] = gsl_sf_bessel_I1_scaled(ki[i]) / I0[i];
  }
};

struct RowWeightsWorker : public RcppParallel::Worker {
  const double* mu1; const double* mu2;
  const double* y1; const double* y2;
  const double* ki; const double* A1;
  double* d11; double* d22; double* d12; double* r1; double* r2;

  RowWeightsWorker(const arma::mat& mu_, const arma::mat& y_,
                   const arma::vec& ki_, const arma::vec& A1_,
                   arma::vec& d11_, arma::vec& d22_, arma::vec& d12_,
                   arma::vec& r1_, arma::vec& r2_)
    : mu1(mu_.colptr(0)), mu2(mu_.colptr(1)),
      y1(y_.colptr(0)), y2(y_.colptr(1)),
      ki(ki_.memptr()), A1(A1_.memptr()),
      d11(d11_.memptr()), d22(d22_.memptr()), d12(d12_.memptr()),
      r1(r1_.memptr()), r2(r2_.memptr()) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      const double k = ki[i], a = A1[i];
      const double aki = a / k, ad = 1.0 - aki - a * a;
      const double u1 = mu1[i] / k, u2 = mu2[i] / k;
      const double u1s = u1 * u1, u2s = u2 * u2;
      d11[i] = ad * u1s + aki * (1.0 - u1s);
      d22[i] = ad * u2s + aki * (1.0 - u2s);
      d12[i] = (ad - aki) * u1 * u2;
      r1[i] = y1[i] - mu1[i] * aki;
      r2[i] = y2[i] - mu2[i] * aki;
    }
  }
};

struct HessianWorker : public RcppParallel::Worker {
  const double* X; const double* d11; const double* d22; const double* d12;
  double* H;
  arma::uword n, p, dim;

  HessianWorker(const arma::mat& X_, const arma::vec& d11_,
                const arma::vec& d22_, const arma::vec& d12_,
                arma::mat& H_)
    : X(X_.memptr()), d11(d11_.memptr()), d22(d22_.memptr()),
      d12(d12_.memptr()), H(H_.memptr()), n(X_.n_rows),
      p(X_.n_cols), dim(H_.n_rows) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t jj = begin; jj < end; ++jj) {
      const arma::uword j = static_cast<arma::uword>(jj);
      const double* xj = X + j * n;
      for (arma::uword k = 0; k <= j; ++k) {
        const double* xk = X + k * n;
        double s11 = 0.0, s22 = 0.0, s12 = 0.0;
        for (arma::uword i = 0; i < n; ++i) {
          const double v = xj[i] * xk[i];
          s11 += v * d11[i]; s22 += v * d22[i]; s12 += v * d12[i];
        }
        const arma::uword pj = p + j, pk = p + k;
        H[j + k * dim] = H[k + j * dim] = -s11;
        H[pj + pk * dim] = H[pk + pj * dim] = -s22;
        H[j + pk * dim] = H[pk + j * dim] = -s12;
        H[k + pj * dim] = H[pj + k * dim] = -s12;
      }
    }
  }
};

inline void row_norm2_with_threads(const arma::mat& z, arma::vec& out,
                                   int ncores) {
  out.set_size(z.n_rows);
  if (ncores <= 1) { row_norm2(z, out); return; }
  RowNorm2Worker worker(z, out);
  RcppParallel::parallelFor(0, static_cast<std::size_t>(z.n_rows),
                            worker, 1000, ncores);
}

inline void compute_I0_with_threads(const arma::vec& ki, arma::vec& I0,
                                    int ncores) {
  if (ncores <= 1) { compute_I0(ki, I0); return; }
  I0.set_size(ki.n_elem);
  BesselI0Worker worker(ki, I0);
  RcppParallel::parallelFor(0, static_cast<std::size_t>(ki.n_elem),
                            worker, 1000, ncores);
}

inline void compute_A1_with_threads(const arma::vec& ki,
                                    const arma::vec& I0,
                                    arma::vec& A1, int ncores) {
  if (ncores <= 1) { compute_A1(ki, I0, A1); return; }
  A1.set_size(ki.n_elem);
  BesselA1Worker worker(ki, I0, A1);
  RcppParallel::parallelFor(0, static_cast<std::size_t>(ki.n_elem),
                            worker, 1000, ncores);
}

inline void compute_row_weights_with_threads(
    const arma::mat& mu, const arma::mat& y, const arma::vec& ki,
    const arma::vec& A1, arma::vec& d11, arma::vec& d22, arma::vec& d12,
    arma::vec& r1, arma::vec& r2, int ncores) {
  if (ncores <= 1) {
    const double* mup = mu.colptr(0); const double* mum = mu.colptr(1);
    const double* yp = y.colptr(0); const double* ym = y.colptr(1);
    for (arma::uword i = 0; i < ki.n_elem; ++i) {
      const double k = ki[i], a = A1[i];
      const double aki = a / k, ad = 1.0 - aki - a * a;
      const double u1 = mup[i] / k, u2 = mum[i] / k;
      const double u1s = u1 * u1, u2s = u2 * u2;
      d11[i] = ad * u1s + aki * (1.0 - u1s);
      d22[i] = ad * u2s + aki * (1.0 - u2s);
      d12[i] = (ad - aki) * u1 * u2;
      r1[i] = yp[i] - mup[i] * aki;
      r2[i] = ym[i] - mum[i] * aki;
    }
    return;
  }
  RowWeightsWorker worker(mu, y, ki, A1, d11, d22, d12, r1, r2);
  RcppParallel::parallelFor(0, static_cast<std::size_t>(ki.n_elem),
                            worker, 1000, ncores);
}

inline void compute_hessian_with_threads(
    const arma::mat& X, const arma::vec& d11, const arma::vec& d22,
    const arma::vec& d12, arma::mat& H, int ncores) {
  H.zeros();
  if (ncores <= 1) {
    const arma::uword n = X.n_rows, p = X.n_cols;
    for (arma::uword j = 0; j < p; ++j) {
      const double* xj = X.colptr(j);
      for (arma::uword k = 0; k <= j; ++k) {
        const double* xk = X.colptr(k);
        double s11 = 0.0, s22 = 0.0, s12 = 0.0;
        for (arma::uword i = 0; i < n; ++i) {
          const double v = xj[i] * xk[i];
          s11 += v * d11[i]; s22 += v * d22[i]; s12 += v * d12[i];
        }
        const arma::uword pj = p + j, pk = p + k;
        H(j, k) = H(k, j) = -s11;
        H(pj, pk) = H(pk, pj) = -s22;
        H(j, pk) = H(pk, j) = -s12;
        H(k, pj) = H(pj, k) = -s12;
      }
    }
    return;
  }
  HessianWorker worker(X, d11, d22, d12, H);
  RcppParallel::parallelFor(0, static_cast<std::size_t>(X.n_cols),
                            worker, 1, ncores);
}

} // namespace

struct AnglesToXYWorker : public RcppParallel::Worker {
  const RcppParallel::RVector<double> y;
  RcppParallel::RMatrix<double> out;
  const bool rads;

  AnglesToXYWorker(
    Rcpp::NumericVector y_,
    Rcpp::NumericMatrix out_,
    bool rads_
  ) : y(y_), out(out_), rads(rads_) {}

  void operator()(std::size_t begin, std::size_t end) {
    if (rads) {
      for (std::size_t i = begin; i < end; ++i) {
        const double theta = y[i];
        out(i, 0) = std::cos(theta);
        out(i, 1) = std::sin(theta);
      }
    } else {
      const double factor = std::acos(-1.0) / 180.0;

      for (std::size_t i = begin; i < end; ++i) {
        const double theta = y[i] * factor;
        out(i, 0) = std::cos(theta);
        out(i, 1) = std::sin(theta);
      }
    }
  }
};

// [[Rcpp::export]]
Rcpp::NumericMatrix angles_to_xy_rcpp(
    Rcpp::NumericVector y,
    bool rads = true,
    int ncores = 1
) {
  if (ncores < 1) {
    Rcpp::stop("ncores must be >= 1.");
  }

  const R_xlen_t n = y.size();
  Rcpp::NumericMatrix out(n, 2);

  // Serial implementation
  if (ncores == 1) {
    const double* yp = REAL(y);
    double* out_cos = REAL(out);
    double* out_sin = out_cos + n;

    if (rads) {
      for (R_xlen_t i = 0; i < n; ++i) {
        const double theta = yp[i];
        out_cos[i] = std::cos(theta);
        out_sin[i] = std::sin(theta);
      }
    } else {
      const double factor = std::acos(-1.0) / 180.0;

      for (R_xlen_t i = 0; i < n; ++i) {
        const double theta = yp[i] * factor;
        out_cos[i] = std::cos(theta);
        out_sin[i] = std::sin(theta);
      }
    }

    return out;
  }

  // Parallel implementation
  AnglesToXYWorker worker(y, out, rads);

  RcppParallel::parallelFor(
    0,
    static_cast<std::size_t>(n),
    worker,
    1000,
    ncores
  );

  return out;
}

// [[Rcpp::export]]
Rcpp::List vm_reg_rcpp_serial(
    const arma::mat& y,
    const arma::mat& x_in,
    Rcpp::Nullable<arma::mat> xnew_ = R_NilValue,
    double tol = 1e-6,
    int maxiters = 500
) {
  const arma::uword n = x_in.n_rows;
  const arma::uword p0 = x_in.n_cols;
  const arma::uword p = p0 + 1;

  if (y.n_rows != n || y.n_cols != 2)
    Rcpp::stop("y must have two columns and the same number of rows as x_in.");

  // Design matrix with intercept.
  arma::mat x(n, p, arma::fill::ones);
  if (p0 > 0)
    x.cols(1, p0) = x_in;

  const arma::mat xt = x.t();

  // Preserve the original initialization.
  arma::mat be = arma::solve(xt * x, xt * y);
  arma::mat mu = x * be;

  arma::vec ki, I0, A1;
  row_norm2(mu, ki);
  ki = arma::sqrt(ki);
  compute_I0(ki, I0);

  double lik1 =
    arma::accu(mu % y) - arma::accu(arma::log(I0) + ki);
  double lik2 = lik1;

  arma::vec d11(n), d22(n), d12(n), r1(n), r2(n);
  arma::mat grad(p, 2, arma::fill::zeros);
  arma::mat H(2 * p, 2 * p, arma::fill::zeros);

  compute_A1(ki, I0, A1);

  int iter = 1;

  while (iter < maxiters) {
    const double* mup = mu.colptr(0);
    const double* mum = mu.colptr(1);
    const double* yp = y.colptr(0);
    const double* ym = y.colptr(1);

    // Serial observation-level residuals and Hessian weights.
    for (arma::uword i = 0; i < n; ++i) {
      const double k = ki[i];
      const double a = A1[i];
      const double aki = a / k;
      const double ad = 1.0 - aki - a * a;

      const double u1 = mup[i] / k;
      const double u2 = mum[i] / k;
      const double u1s = u1 * u1;
      const double u2s = u2 * u2;

      d11[i] = ad * u1s + aki * (1.0 - u1s);
      d22[i] = ad * u2s + aki * (1.0 - u2s);
      d12[i] = (ad - aki) * u1 * u2;

      r1[i] = yp[i] - mup[i] * aki;
      r2[i] = ym[i] - mum[i] * aki;
    }

    grad.col(0) = xt * r1;
    grad.col(1) = xt * r2;

    // Serial Hessian construction.
    for (arma::uword j = 0; j < p; ++j) {
      const double* xj = x.colptr(j);

      for (arma::uword k = 0; k <= j; ++k) {
        const double* xk = x.colptr(k);

        const arma::uword pj = p + j;
        const arma::uword pk = p + k;

        double s11 = 0.0;
        double s22 = 0.0;
        double s12 = 0.0;

        for (arma::uword i = 0; i < n; ++i) {
          const double v = xj[i] * xk[i];
          s11 += v * d11[i];
          s22 += v * d22[i];
          s12 += v * d12[i];
        }

        H(j, k) = H(k, j) = -s11;
        H(pj, pk) = H(pk, pj) = -s22;
        H(j, pk) = H(pk, j) = -s12;
        H(k, pj) = H(pj, k) = -s12;
      }
    }

    arma::vec step = arma::solve(H, arma::vectorise(grad));
    arma::vec be_vec = arma::vectorise(be);
    be_vec -= step;
    be = arma::reshape(be_vec, p, 2);

    mu = x * be;
    row_norm2(mu, ki);
    ki = arma::sqrt(ki);
    compute_I0(ki, I0);

    lik2 =
      arma::accu(mu % y) - arma::accu(arma::log(I0) + ki);

    if ((lik2 - lik1) <= tol)
      break;

    lik1 = lik2;
    compute_A1(ki, I0, A1);
    ++iter;
  }

  SEXP est_out = R_NilValue;

  if (xnew_.isNotNull()) {
    arma::mat xnew = Rcpp::as<arma::mat>(xnew_);
    arma::mat Xn(
        xnew.n_rows, xnew.n_cols + 1, arma::fill::ones
    );

    if (xnew.n_cols > 0)
      Xn.cols(1, xnew.n_cols) = xnew;

    arma::mat est = Xn * be;
    arma::vec nm;

    row_norm2(est, nm);
    nm = arma::sqrt(nm);
    est.each_col() /= nm;

    est_out = Rcpp::wrap(est);
  }

  return Rcpp::List::create(
    Rcpp::_["be"] = be,
    Rcpp::_["loglik"] = lik2 - n * std::log(2.0 * M_PI),
    Rcpp::_["ki"] = Rcpp::NumericVector(ki.begin(), ki.end()),
    Rcpp::_["A1"] = Rcpp::NumericVector(A1.begin(), A1.end()),
    Rcpp::_["est"] = est_out
  );
}


// [[Rcpp::export]]
Rcpp::List vm_reg_rcpp_parallel(
    const arma::mat& y,
    const arma::mat& x_in,
    Rcpp::Nullable<arma::mat> xnew_ = R_NilValue,
    double tol = 1e-6,
    int maxiters = 500,
    int ncores = 1
) {
  if (ncores < 1)
    Rcpp::stop("ncores must be >= 1.");

  const arma::uword n = x_in.n_rows;
  const arma::uword p0 = x_in.n_cols;
  const arma::uword p = p0 + 1;

  if (y.n_rows != n || y.n_cols != 2)
    Rcpp::stop("y must have two columns and the same number of rows as x_in.");

  // Design matrix with intercept.
  arma::mat x(n, p, arma::fill::ones);
  if (p0 > 0)
    x.cols(1, p0) = x_in;

  const arma::mat xt = x.t();

  // Preserve the original initialization.
  arma::mat be = arma::solve(xt * x, xt * y);
  arma::mat mu = x * be;

  arma::vec ki, I0, A1;
  row_norm2_with_threads(mu, ki, ncores);
  ki = arma::sqrt(ki);
  compute_I0_with_threads(ki, I0, ncores);

  double lik1 =
    arma::accu(mu % y) - arma::accu(arma::log(I0) + ki);
  double lik2 = lik1;

  arma::vec d11(n), d22(n), d12(n), r1(n), r2(n);
  arma::mat grad(p, 2, arma::fill::zeros);
  arma::mat H(2 * p, 2 * p, arma::fill::zeros);

  compute_A1_with_threads(ki, I0, A1, ncores);

  int iter = 1;

  while (iter < maxiters) {
    // Observation-level calculations.
    compute_row_weights_with_threads(
      mu, y, ki, A1,
      d11, d22, d12, r1, r2, ncores
    );

    // Gradient.
    grad.col(0) = xt * r1;
    grad.col(1) = xt * r2;

    // Hessian.
    compute_hessian_with_threads(
      x, d11, d22, d12, H, ncores
    );

    // Newton update.
    arma::vec step = arma::solve(H, arma::vectorise(grad));
    arma::vec be_vec = arma::vectorise(be);
    be_vec -= step;
    be = arma::reshape(be_vec, p, 2);

    // Updated linear predictor and likelihood.
    mu = x * be;
    row_norm2_with_threads(mu, ki, ncores);
    ki = arma::sqrt(ki);
    compute_I0_with_threads(ki, I0, ncores);

    lik2 =
      arma::accu(mu % y) - arma::accu(arma::log(I0) + ki);

    if ((lik2 - lik1) <= tol)
      break;

    lik1 = lik2;
    compute_A1_with_threads(ki, I0, A1, ncores);
    ++iter;
  }

  // Optional predictions.
  SEXP est_out = R_NilValue;

  if (xnew_.isNotNull()) {
    arma::mat xnew = Rcpp::as<arma::mat>(xnew_);
    arma::mat Xn(
        xnew.n_rows, xnew.n_cols + 1, arma::fill::ones
    );

    if (xnew.n_cols > 0)
      Xn.cols(1, xnew.n_cols) = xnew;

    arma::mat est = Xn * be;
    arma::vec nm;

    row_norm2_with_threads(est, nm, ncores);
    nm = arma::sqrt(nm);
    est.each_col() /= nm;

    est_out = Rcpp::wrap(est);
  }

  return Rcpp::List::create(
    Rcpp::_["be"] = be,
    Rcpp::_["loglik"] = lik2 - n * std::log(2.0 * M_PI),
    Rcpp::_["ki"] = Rcpp::NumericVector(ki.begin(), ki.end()),
    Rcpp::_["A1"] = Rcpp::NumericVector(A1.begin(), A1.end()),
    Rcpp::_["est"] = est_out
  );
}


// [[Rcpp::export]]
double vm_reg_loglik_cpp(
    const arma::mat& y,
    const arma::mat& x_in,
    double tol = 1e-6,
    int maxiters = 500) {

  const arma::uword n = x_in.n_rows;
  const arma::uword p0 = x_in.n_cols;
  const arma::uword p = p0 + 1;

  // Design matrix with intercept.
  arma::mat x(n, p, arma::fill::ones);
  if (p0 > 0) {
    x.cols(1, p0) = x_in;
  }

  const arma::mat xt = x.t();

  // Preserve the initialization of the reference implementation.
  arma::mat be = arma::solve(xt * x, xt * y);
  arma::mat mu = x * be;

  arma::vec ki, I0, A1;
  row_norm2(mu, ki);
  ki = arma::sqrt(ki);
  compute_I0(ki, I0);

  double lik1 =
    arma::accu(mu % y) - arma::accu(arma::log(I0) + ki);
  double lik2 = lik1;

  arma::vec d11(n), d22(n), d12(n), r1(n), r2(n);
  arma::mat grad(p, 2, arma::fill::zeros);
  arma::mat H(2 * p, 2 * p, arma::fill::zeros);

  compute_A1(ki, I0, A1);

  int iter = 1;

  while (iter < maxiters) {

    const double* mup = mu.colptr(0);
    const double* mum = mu.colptr(1);
    const double* yp = y.colptr(0);
    const double* ym = y.colptr(1);

    // Row-level Hessian weights and gradient residuals.
    for (arma::uword i = 0; i < n; ++i) {
      const double k = ki[i];
      const double a = A1[i];
      const double aki = a / k;
      const double ad = 1.0 - aki - a * a;

      const double u1 = mup[i] / k;
      const double u2 = mum[i] / k;
      const double u1s = u1 * u1;
      const double u2s = u2 * u2;

      d11[i] = ad * u1s + aki * (1.0 - u1s);
      d22[i] = ad * u2s + aki * (1.0 - u2s);
      d12[i] = (ad - aki) * u1 * u2;

      r1[i] = yp[i] - mup[i] * aki;
      r2[i] = ym[i] - mum[i] * aki;
    }

    grad.col(0) = xt * r1;
    grad.col(1) = xt * r2;

    // Construct the Hessian using fused loops.
    for (arma::uword j = 0; j < p; ++j) {
      const double* xj = x.colptr(j);

      for (arma::uword k = 0; k <= j; ++k) {
        const double* xk = x.colptr(k);

        const arma::uword pj = p + j;
        const arma::uword pk = p + k;

        double s11 = 0.0;
        double s22 = 0.0;
        double s12 = 0.0;

        for (arma::uword i = 0; i < n; ++i) {
          const double v = xj[i] * xk[i];

          s11 += v * d11[i];
          s22 += v * d22[i];
          s12 += v * d12[i];
        }

        H(j, k) = H(k, j) = -s11;
        H(pj, pk) = H(pk, pj) = -s22;
        H(j, pk) = H(pk, j) = -s12;
        H(k, pj) = H(pj, k) = -s12;
      }
    }

    arma::vec step = arma::solve(H, arma::vectorise(grad));
    arma::vec be_vec = arma::vectorise(be);
    be_vec -= step;
    be = arma::reshape(be_vec, p, 2);

    mu = x * be;
    row_norm2(mu, ki);
    ki = arma::sqrt(ki);
    compute_I0(ki, I0);

    lik2 = arma::accu(mu % y) -
      arma::accu(arma::log(I0) + ki);

    if ((lik2 - lik1) <= tol) {
      break;
    }

    lik1 = lik2;
    compute_A1(ki, I0, A1);
    ++iter;
  }

  return lik2 - n * std::log(2.0 * M_PI);
}

// [[Rcpp::export]]
double vm_reg_loglik_export(
    const arma::mat& y,
    const arma::mat& x,
    double tol = 1e-6,
    int maxiters = 500
) {
  return vm_reg_loglik_cpp(y, x, tol, maxiters);
}

// [[Rcpp::export]]
double perm_reg0_cpp(
    const arma::mat& y,
    const arma::mat& x,
    double lik0,
    double stat,
    double tol = 1e-6,
    int maxiters = 500,
    int R = 999
) {
  RNGScope scope;

  const int n = y.n_rows;

  // Call base R's sample.int(), preserving its sampling algorithm.
  Environment base = Environment::base_env();
  Function sample_int = base["sample.int"];

  arma::mat yb(n, y.n_cols);

  int exceed = 0;

  for (int r = 0; r < R; ++r) {

    // Exactly the same sampling operation as sample.int(n, n) in R.
    IntegerVector idx = sample_int(n, n);

    // Reorder response rows; convert R's 1-based indices to C++ indices.
    for (arma::uword i = 0; i < y.n_rows; ++i) {
      for (arma::uword j = 0; j < y.n_cols; ++j) {
        yb(i, j) = y(idx[i] - 1, j);
      }
    }

    double ll = vm_reg_loglik_cpp(yb, x, tol, maxiters);

    if (ll - lik0 >= stat)
      ++exceed;
  }

  return (exceed + 1.0) / (R + 1.0);
}


double vm_reg_loglik_precomp_cpp(
    const arma::mat& y,
    const arma::mat& X,
    const arma::mat& Xt,
    const arma::mat& XtX_inv,
    double tol,
    int maxiters,
    bool parallel_safe_bessel = false
) {
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;

  arma::mat be = XtX_inv * (Xt * y);
  arma::mat mu = X * be;

  arma::vec ki, I0, A1;
  row_norm2(mu, ki);
  ki = arma::sqrt(ki);
  if (parallel_safe_bessel) compute_I0_parallel_safe(ki, I0);
  else compute_I0(ki, I0);

  double lik1 =
    arma::accu(mu % y) - arma::accu(arma::log(I0) + ki);
  double lik2 = lik1;

  arma::vec d11(n), d22(n), d12(n), r1(n), r2(n);
  arma::mat grad(p, 2, arma::fill::zeros);
  arma::mat H(2 * p, 2 * p, arma::fill::zeros);

  if (parallel_safe_bessel) compute_A1_parallel_safe(ki, I0, A1);
  else compute_A1(ki, I0, A1);

  for (int iter = 1; iter < maxiters; ++iter) {
    const double* mup = mu.colptr(0);
    const double* mum = mu.colptr(1);
    const double* yp = y.colptr(0);
    const double* ym = y.colptr(1);

    for (arma::uword i = 0; i < n; ++i) {
      const double k = ki[i];
      const double a = A1[i];
      const double aki = a / k;
      const double ad = 1.0 - aki - a * a;

      const double u1 = mup[i] / k;
      const double u2 = mum[i] / k;
      const double u1s = u1 * u1;
      const double u2s = u2 * u2;

      d11[i] = ad * u1s + aki * (1.0 - u1s);
      d22[i] = ad * u2s + aki * (1.0 - u2s);
      d12[i] = (ad - aki) * u1 * u2;

      r1[i] = yp[i] - mup[i] * aki;
      r2[i] = ym[i] - mum[i] * aki;
    }

    grad.col(0) = Xt * r1;
    grad.col(1) = Xt * r2;

    for (arma::uword j = 0; j < p; ++j) {
      const double* xj = X.colptr(j);

      for (arma::uword k = 0; k <= j; ++k) {
        const double* xk = X.colptr(k);
        const arma::uword pj = p + j;
        const arma::uword pk = p + k;

        double s11 = 0.0, s22 = 0.0, s12 = 0.0;

        for (arma::uword i = 0; i < n; ++i) {
          const double v = xj[i] * xk[i];
          s11 += v * d11[i];
          s22 += v * d22[i];
          s12 += v * d12[i];
        }

        H(j, k) = H(k, j) = -s11;
        H(pj, pk) = H(pk, pj) = -s22;
        H(j, pk) = H(pk, j) = -s12;
        H(k, pj) = H(pj, k) = -s12;
      }
    }

    arma::vec step = arma::solve(H, arma::vectorise(grad));
    arma::vec bv = arma::vectorise(be);
    bv -= step;
    be = arma::reshape(bv, p, 2);

    mu = X * be;
    row_norm2(mu, ki);
    ki = arma::sqrt(ki);
    if (parallel_safe_bessel) compute_I0_parallel_safe(ki, I0);
    else compute_I0(ki, I0);

    lik2 = arma::accu(mu % y) -
      arma::accu(arma::log(I0) + ki);

    if ((lik2 - lik1) <= tol)
      break;

    lik1 = lik2;
    if (parallel_safe_bessel) compute_A1_parallel_safe(ki, I0, A1);
    else compute_A1(ki, I0, A1);
  }

  return lik2 - n * std::log(2.0 * M_PI);
}


// [[Rcpp::export]]
double perm_reg0_serial_idx_cpp(
    const arma::mat& y,
    const arma::mat& x,
    double lik0,
    double stat,
    const arma::imat& perm_idx,
    double tol = 1e-6,
    int maxiters = 500
) {
  const arma::uword n = y.n_rows;
  const int R = static_cast<int>(perm_idx.n_cols);

  if (y.n_cols != 2 || x.n_rows != n || perm_idx.n_rows != n)
    Rcpp::stop("Incompatible dimensions.");
  if (R == 0)
    return 1.0;

  // Match vm_reg_loglik_cpp(): include an intercept.
  arma::mat X(n, x.n_cols + 1, arma::fill::ones);
  if (x.n_cols > 0)
    X.cols(1, x.n_cols) = x;

  const arma::mat Xt = X.t();
  const arma::mat XtX_inv = arma::pinv(Xt * X);
  arma::mat yb(n, y.n_cols);
  int exceed = 0;

  for (int r = 0; r < R; ++r) {
    for (arma::uword i = 0; i < n; ++i) {
      const arma::uword j = static_cast<arma::uword>(perm_idx(i, r) - 1);
      if (j >= n)
        Rcpp::stop("perm_idx contains an out-of-range index.");
      for (arma::uword k = 0; k < y.n_cols; ++k)
        yb(i, k) = y(j, k);
    }

    // Serial path uses the same Boost-based scaled-Bessel implementation.
    const double ll = vm_reg_loglik_precomp_cpp(
      yb, X, Xt, XtX_inv, tol, maxiters, false);

    if (ll - lik0 >= stat)
      ++exceed;
  }

  return (exceed + 1.0) / (static_cast<double>(R) + 1.0);
}


// RcppParallel worker for the test-0 permutation distribution.
// Each permutation is evaluated independently and writes to its own flag.
struct PermReg0Worker : public RcppParallel::Worker {
  const arma::mat& y;
  const arma::mat& X;
  const arma::mat& Xt;
  const arma::mat& XtX_inv;
  const arma::imat& perm_idx;
  const double lik0;
  const double stat;
  const double tol;
  const int maxiters;
  int* exceed_flags;
  const arma::uword n;

  PermReg0Worker(const arma::mat& y_,
                 const arma::mat& X_,
                 const arma::mat& Xt_,
                 const arma::mat& XtX_inv_,
                 const arma::imat& perm_idx_,
                 double lik0_,
                 double stat_,
                 double tol_,
                 int maxiters_,
                 int* exceed_flags_)
    : y(y_), X(X_), Xt(Xt_), XtX_inv(XtX_inv_),
      perm_idx(perm_idx_), lik0(lik0_), stat(stat_), tol(tol_),
      maxiters(maxiters_), exceed_flags(exceed_flags_), n(y_.n_rows) {}

  void operator()(std::size_t begin, std::size_t end) {
    arma::mat yb(n, y.n_cols);
    for (std::size_t rr = begin; rr < end; ++rr) {
      const arma::uword r = static_cast<arma::uword>(rr);
      for (arma::uword i = 0; i < n; ++i) {
        const arma::uword j = static_cast<arma::uword>(perm_idx(i, r) - 1);
        for (arma::uword k = 0; k < y.n_cols; ++k)
          yb(i, k) = y(j, k);
      }

      const double ll = vm_reg_loglik_precomp_cpp(
        yb, X, Xt, XtX_inv, tol, maxiters, true);
      exceed_flags[rr] = (ll - lik0 >= stat) ? 1 : 0;
    }
  }
};

// [[Rcpp::export]]
double perm_reg0_parallel_cpp(
    const arma::mat& y,
    const arma::mat& x,
    double lik0,
    double stat,
    const arma::imat& perm_idx,
    double tol = 1e-6,
    int maxiters = 500,
    int ncores = 1
) {
  const arma::uword n = y.n_rows;
  const int R = static_cast<int>(perm_idx.n_cols);

  if (y.n_cols != 2 || x.n_rows != n || perm_idx.n_rows != n)
    Rcpp::stop("Incompatible dimensions.");
  if (ncores < 1)
    Rcpp::stop("ncores must be >= 1.");
  if (R == 0)
    return 1.0;

  // Match vm_reg_loglik_cpp(): add an intercept to the design matrix.
  arma::mat X(n, x.n_cols + 1, arma::fill::ones);
  if (x.n_cols > 0)
    X.cols(1, x.n_cols) = x;

  const arma::mat Xt = X.t();
  const arma::mat XtX_inv = arma::pinv(Xt * X);
  std::vector<int> exceed_flags(static_cast<std::size_t>(R), 0);

  PermReg0Worker worker(y, X, Xt, XtX_inv, perm_idx, lik0, stat,
                        tol, maxiters, exceed_flags.data());
  RcppParallel::parallelFor(
    0, static_cast<std::size_t>(R), worker, 1, ncores);

  int exceed = 0;
  for (int r = 0; r < R; ++r)
    exceed += exceed_flags[static_cast<std::size_t>(r)];

  return (exceed + 1.0) / (static_cast<double>(R) + 1.0);
}

// [[Rcpp::export]]
double perm_reg1_serial_fast_cpp(
    const arma::mat& y,
    const arma::mat& xsel,
    const arma::mat& xf,
    const arma::mat& u,
    double stat,
    const arma::imat& perm_idx,
    double tol = 1e-6,
    int maxiters = 500
) {
  const arma::uword n = y.n_rows;
  const int R = static_cast<int>(perm_idx.n_cols);

  if (y.n_cols != 2 || u.n_rows != n || u.n_cols != 2 ||
      xsel.n_rows != n || xf.n_rows != n ||
      perm_idx.n_rows != n) {
    Rcpp::stop("Incompatible dimensions.");
  }

  // Build each design matrix once, including the intercept.
  arma::mat X0(n, xsel.n_cols + 1, arma::fill::ones);
  if (xsel.n_cols > 0)
    X0.cols(1, xsel.n_cols) = xsel;

  arma::mat X1(n, xf.n_cols + 1, arma::fill::ones);
  if (xf.n_cols > 0)
    X1.cols(1, xf.n_cols) = xf;

  const arma::mat Xt0 = X0.t();
  const arma::mat Xt1 = X1.t();

  // Precompute inverse cross-products.
  // pinv is used to accommodate possible rank-deficient designs.
  const arma::mat XtX0_inv = arma::pinv(Xt0 * X0);
  const arma::mat XtX1_inv = arma::pinv(Xt1 * X1);

  // Transform the response into the reduced-model frame once.
  arma::vec e1(n), e2(n);

  for (arma::uword i = 0; i < n; ++i) {
    e1[i] = y(i, 0) * u(i, 0) + y(i, 1) * u(i, 1);
    e2[i] = y(i, 1) * u(i, 0) - y(i, 0) * u(i, 1);
  }

  int exceed = 0;
  arma::mat yb(n, 2);

  for (int r = 0; r < R; ++r) {
    for (arma::uword i = 0; i < n; ++i) {
      const arma::uword j =
        static_cast<arma::uword>(perm_idx(i, r) - 1);

      const double a = e1[j];
      const double b = e2[j];

      yb(i, 0) = a * u(i, 0) - b * u(i, 1);
      yb(i, 1) = a * u(i, 1) + b * u(i, 0);
    }

    const double ll_full = vm_reg_loglik_precomp_cpp(
      yb, X1, Xt1, XtX1_inv, tol, maxiters
    );

    const double ll_reduced = vm_reg_loglik_precomp_cpp(
      yb, X0, Xt0, XtX0_inv, tol, maxiters
    );

    if (ll_full - ll_reduced >= stat)
      ++exceed;
  }

  return (exceed + 1.0) / (R + 1.0);
}


// RcppParallel worker: each permutation writes to its own flag, so there
// are no shared-counter races. All matrices/vectors are read-only.
struct PermReg1Worker : public RcppParallel::Worker {
  const arma::mat& y;
  const arma::mat& X0;
  const arma::mat& Xt0;
  const arma::mat& XtX0_inv;
  const arma::mat& X1;
  const arma::mat& Xt1;
  const arma::mat& XtX1_inv;
  const arma::mat& u;
  const arma::vec& e1;
  const arma::vec& e2;
  const arma::imat& perm_idx;
  const double stat;
  const double tol;
  const int maxiters;
  int* exceed_flags;
  const arma::uword n;

  PermReg1Worker(const arma::mat& y_,
                 const arma::mat& X0_,
                 const arma::mat& Xt0_,
                 const arma::mat& XtX0_inv_,
                 const arma::mat& X1_,
                 const arma::mat& Xt1_,
                 const arma::mat& XtX1_inv_,
                 const arma::mat& u_,
                 const arma::vec& e1_,
                 const arma::vec& e2_,
                 const arma::imat& perm_idx_,
                 double stat_,
                 double tol_,
                 int maxiters_,
                 int* exceed_flags_)
    : y(y_), X0(X0_), Xt0(Xt0_), XtX0_inv(XtX0_inv_),
      X1(X1_), Xt1(Xt1_), XtX1_inv(XtX1_inv_), u(u_),
      e1(e1_), e2(e2_), perm_idx(perm_idx_), stat(stat_),
      tol(tol_), maxiters(maxiters_), exceed_flags(exceed_flags_),
      n(y_.n_rows) {}

  void operator()(std::size_t begin, std::size_t end) {
    arma::mat yb(n, 2);

    for (std::size_t rr = begin; rr < end; ++rr) {
      const arma::uword r = static_cast<arma::uword>(rr);

      for (arma::uword i = 0; i < n; ++i) {
        // perm_idx stores R-style (1-based) indices.
        const arma::uword j =
          static_cast<arma::uword>(perm_idx(i, r) - 1);

        const double a = e1[j];
        const double b = e2[j];

        yb(i, 0) = a * u(i, 0) - b * u(i, 1);
        yb(i, 1) = a * u(i, 1) + b * u(i, 0);
      }

      const double ll_full = vm_reg_loglik_precomp_cpp(
        yb, X1, Xt1, XtX1_inv, tol, maxiters, true);

      const double ll_reduced = vm_reg_loglik_precomp_cpp(
        yb, X0, Xt0, XtX0_inv, tol, maxiters, true);

      exceed_flags[rr] = (ll_full - ll_reduced >= stat) ? 1 : 0;
    }
  }
};

// [[Rcpp::export]]
double perm_reg1_parallel_fast_cpp(
    const arma::mat& y,
    const arma::mat& xsel,
    const arma::mat& xf,
    const arma::mat& u,
    double stat,
    const arma::imat& perm_idx,
    double tol = 1e-6,
    int maxiters = 500,
    int ncores = 1
) {
  const arma::uword n = y.n_rows;
  const int R = static_cast<int>(perm_idx.n_cols);

  if (y.n_cols != 2 || u.n_rows != n || u.n_cols != 2 ||
      xsel.n_rows != n || xf.n_rows != n ||
      perm_idx.n_rows != n) {
    Rcpp::stop("Incompatible dimensions.");
  }
  if (ncores < 1)
    Rcpp::stop("ncores must be >= 1.");

  arma::mat X0(n, xsel.n_cols + 1, arma::fill::ones);
  if (xsel.n_cols > 0)
    X0.cols(1, xsel.n_cols) = xsel;

  arma::mat X1(n, xf.n_cols + 1, arma::fill::ones);
  if (xf.n_cols > 0)
    X1.cols(1, xf.n_cols) = xf;

  const arma::mat Xt0 = X0.t();
  const arma::mat Xt1 = X1.t();
  const arma::mat XtX0_inv = arma::pinv(Xt0 * X0);
  const arma::mat XtX1_inv = arma::pinv(Xt1 * X1);

  arma::vec e1(n), e2(n);
  for (arma::uword i = 0; i < n; ++i) {
    e1[i] = y(i, 0) * u(i, 0) + y(i, 1) * u(i, 1);
    e2[i] = y(i, 1) * u(i, 0) - y(i, 0) * u(i, 1);
  }

  if (R == 0)
    return 1.0;

  std::vector<int> exceed_flags(static_cast<std::size_t>(R), 0);

  // The final argument requests the number of worker threads for this call.
  PermReg1Worker worker(
      y, X0, Xt0, XtX0_inv, X1, Xt1, XtX1_inv, u, e1, e2,
      perm_idx, stat, tol, maxiters, exceed_flags.data());

  RcppParallel::parallelFor(
    0, static_cast<std::size_t>(R), worker, 1, ncores);

  int exceed = 0;
  for (int r = 0; r < R; ++r)
    exceed += exceed_flags[static_cast<std::size_t>(r)];

  return (exceed + 1.0) / (static_cast<double>(R) + 1.0);
}

