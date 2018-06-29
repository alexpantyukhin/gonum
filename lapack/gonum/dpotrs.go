// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

func (Implementation) Dpotrs(uplo blas.Uplo, n, nrhs int, a []float64, lda int, b []float64, ldb int) {
	if uplo != blas.Upper && uplo != blas.Lower {
		panic(badUplo)
	}
	checkMatrix(n, n, a, lda)
	checkMatrix(n, nrhs, b, ldb)

	if n == 0 || nrhs == 0 {
		return
	}

	bi := blas64.Implementation()
	if uplo == blas.Upper {
		// Solve U^T * U * X = B where U is stored in the upper triangle of A.

		// Solve U^T * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Upper, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		// Solve U * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Upper, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
	} else {
		// Solve L * L^T * X = B where L is stored in the lower triangle of A.

		// Solve L * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
		// Solve L^T * X = B, overwriting B with X.
		bi.Dtrsm(blas.Left, blas.Lower, blas.Trans, blas.NonUnit, n, nrhs, 1, a, lda, b, ldb)
	}
}
