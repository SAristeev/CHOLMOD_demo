#include "cholmod.h"
#include "cholmod_function.h"
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "CHOLMOD_demo.h"

void cholmod_test(FILE* inA, FILE* inB, int SOLVER_VER, int CUDA_support, int PRINT_DEFAULT, int PRINT_TIME) {
    double start, stop, elapsedTime;
    cholmod_sparse* A;
    cholmod_dense* X = NULL, * B, * W, * R = NULL;
    cholmod_factor* L;
    double* Bx, * Xx, * Rx;
    double resid, resid2, t, ta, tf, ts, tot, anorm, bnorm, rcond, anz, xnorm, rnorm, rnorm2,
        axbnorm, fl;
    double one[2], zero[2], minusone[2], beta[2], xlnz;
    int n, isize, xsize, xtype, s, ss, lnz;
    int L_is_super;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    cholmod_common* cm = (cholmod_common*)malloc(sizeof(cholmod_common));
    cholmod_l_start(cm);
    if (CUDA_support) {
        cm->useGPU = 1;
    }

    /* ---------------------------------------------------------------------- */
    /* create basic scalars */
    /* ---------------------------------------------------------------------- */

    zero[0] = 0;
    zero[1] = 0;
    one[0] = 1;
    one[1] = 0;
    minusone[0] = -1;
    minusone[1] = 0;
    beta[0] = 1e-6;
    beta[1] = 0;

    /* ---------------------------------------------------------------------- */
    /* read matrix A */
    /* ---------------------------------------------------------------------- */

    A = cholmod_l_read_sparse(inA, cm);

    xtype = A->xtype;
    anorm = 1;
    n = A->nrow;
#ifndef NMATRIXOPS
    anorm = cholmod_l_norm_sparse(A, 0, cm);
    printf("norm (A,inf) = %g\n", anorm);
    printf("norm (A,1)   = %g\n", cholmod_l_norm_sparse(A, 1, cm));
#endif
    cholmod_l_print_sparse(A, "A", cm);

    /* ---------------------------------------------------------------------- */
    /* read vector B */
    /* ---------------------------------------------------------------------- */
    B = cholmod_l_ones(A->nrow, 1, A->xtype, cm);   /* b = ones(n,1) */
    if (0){
        B = cholmod_l_read_dense(inB, cm);
    }
    cholmod_l_print_dense(B, "B", cm);
    Bx = B->x;
    bnorm = 1;
#ifndef NMATRIXOPS
    bnorm = cholmod_l_norm_dense(B, 0, cm);	/* max norm */
    printf("bnorm %g\n", bnorm);
#endif

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    start = second();
    t = CPUTIME;
    L = cholmod_l_analyze(A, cm);
    ta = CPUTIME - t;
    ta = MAX(ta, 0);
    
    printf("Analyze: flop %g lnz %g\n", cm->fl, cm->lnz);

    if (A->stype == 0)
    {
        printf("Factorizing A*A'+beta*I\n");
        t = CPUTIME;
        cholmod_l_factorize_p(A, beta, NULL, 0, L, cm);
        tf = CPUTIME - t;
        tf = MAX(tf, 0);
    }
    else
    {
        printf("Factorizing A\n");
        t = CPUTIME;
        cholmod_l_factorize(A, L, cm);
        tf = CPUTIME - t;
        tf = MAX(tf, 0);
    }

    cholmod_l_print_factor(L, "L", cm);

    /* determine the # of integers's and reals's in L.  See cholmod_free */
    if (L->is_super)
    {
        s = L->nsuper + 1;
        xsize = L->xsize;
        ss = L->ssize;
        isize = n	/* L->Perm */
            + n	/* L->ColCount, nz in each column of 'pure' L */
            + s	/* L->pi, column pointers for L->s */
            + s	/* L->px, column pointers for L->x */
            + s	/* L->super, starting column index of each supernode */
            + ss;	/* L->s, the pattern of the supernodes */
    }
    else
    {
        /* this space can increase if you change parameters to their non-
         * default values (cm->final_pack, for example). */
        lnz = L->nzmax;
        xsize = lnz;
        isize =
            n	/* L->Perm */
            + n	/* L->ColCount, nz in each column of 'pure' L */
            + n + 1	/* L->p, column pointers */
            + lnz	/* L->i, integer row indices */
            + n	/* L->nz, nz in each column of L */
            + n + 2	/* L->next, link list */
            + n + 2;	/* L->prev, link list */
    }
    rcond = cholmod_l_rcond(L, cm);
    L_is_super = L->is_super;

    /* ---------------------------------------------------------------------- */
    /* solve */
    /* ---------------------------------------------------------------------- */

    
    if (SOLVER_VER == 0) {
        /* basic solve */
        t = CPUTIME;
        X = cholmod_l_solve(CHOLMOD_A, L, B, cm);
        ts = CPUTIME - t;
        ts = MAX(ts, 0);
    }

    else if (SOLVER_VER == 1) {
        /* solve with reused workspace */
        cholmod_dense* Ywork = NULL, * Ework = NULL;
        t = CPUTIME;
        cholmod_l_solve2(CHOLMOD_A, L, B, NULL, &X, NULL,
            &Ywork, &Ework, cm);
        cholmod_l_free_dense(&Ywork, cm);
        cholmod_l_free_dense(&Ework, cm);
        ts = CPUTIME - t;
    }
//    else {
//        /* solve with reused workspace and sparse Bset */
//        cholmod_dense* Ywork = NULL, * Ework = NULL;
//        cholmod_dense* X2 = NULL, * B2 = NULL;
//        cholmod_sparse* Bset, * Xset = NULL;
//        int* Bsetp, * Bseti, * Xsetp, * Xseti, xlen, j, k, * Lnz;
//        double* X1x, * X2x, * B2x, err;
//        FILE* timelog = fopen("timelog.m", "w");
//        if (timelog) fprintf(timelog, "results = [\n");
//
//        B2 = cholmod_zeros(n, 1, xtype, cm);
//        B2x = B2->x;
//
//        Bset = cholmod_allocate_sparse(n, 1, 1, FALSE, TRUE, 0,
//            CHOLMOD_PATTERN, cm);
//        Bsetp = Bset->p;
//        Bseti = Bset->i;
//        Bsetp[0] = 0;     /* nnz(B) is 1 (it can be anything) */
//        Bsetp[1] = 1;
//        resid = 0;
//
//        for (int i = 0; i < n; i++)
//        {
//            /* B (i) is nonzero, all other entries are ignored
//               (implied to be zero) */
//            Bseti[0] = i;
//            if (xtype == CHOLMOD_REAL)
//            {
//                B2x[i] = Bx[i];
//            }
//
//            /* first get the entire solution, to compare against */
//            cholmod_solve2(CHOLMOD_A, L, B2, NULL, &X, NULL,
//                &Ywork, &Ework, cm);
//
//            /* now get the sparse solutions; this will change L from
//               supernodal to simplicial */
//
//            if (i == 0)
//            {
//                /* first solve can be slower because it has to allocate
//                   space for X2, Xset, etc, and change L.
//                   So don't time it */
//                cholmod_solve2(CHOLMOD_A, L, B2, Bset, &X2, &Xset,
//                    &Ywork, &Ework, cm);
//            }
//
//            t = CPUTIME;
//            /* solve Ax=b but only to get x(i).
//                b is all zero except for b(i).
//                This takes O(xlen) time */
//            cholmod_solve2(CHOLMOD_A, L, B2, Bset, &X2, &Xset,
//                &Ywork, &Ework, cm);
//            t = CPUTIME - t;
//            t = MAX(t, 0);
//            ts = CPUTIME - t;
//            
//
//            /* check the solution and log the time */
//            Xsetp = Xset->p;
//            Xseti = Xset->i;
//            xlen = Xsetp[1];
//            X1x = X->x;
//            X2x = X2->x;
//            Lnz = L->nz;
//
//            /*
//            printf ("\ni %d xlen %d  (%p %p)\n", i, xlen, X1x, X2x) ;
//            */
//
//            if (xtype == CHOLMOD_REAL)
//            {
//                fl = 2 * xlen;
//                for (k = 0; k < xlen; k++)
//                {
//                    j = Xseti[k];
//                    fl += 4 * Lnz[j];
//                    err = X1x[j] - X2x[j];
//                    err = ABS(err);
//                    resid = MAX(resid, err);
//                }
//            }
//            if (timelog) fprintf(timelog, "%g %g %g %g\n",
//                (double)i, (double)xlen, fl, t);
//
//            /* clear B for the next test */
//            if (xtype == CHOLMOD_REAL)
//            {
//                B2x[i] = 0;
//            }
//        }
//
//        if (timelog)
//        {
//            fprintf(timelog, "] ; resid = %g ;\n", resid);
//            fprintf(timelog, "lnz = %g ;\n", cm->lnz);
//            //fprintf(timelog, "t = %g ;   %% dense solve time\n", ts[2]);
//            fclose(timelog);
//        }
//
//#ifndef NMATRIXOPS
//        resid = resid / cholmod_norm_dense(X, 1, cm);
//#endif
//
//        cholmod_free_dense(&Ywork, cm);
//        cholmod_free_dense(&Ework, cm);
//        cholmod_free_dense(&X2, cm);
//        cholmod_free_dense(&B2, cm);
//        cholmod_free_sparse(&Xset, cm);
//        cholmod_free_sparse(&Bset, cm);
//
//    }
    stop = second();
    tot = ta + tf + ts;
    /* ------------------------------------------------------------------ */
    /* compute the residual */
    /* ------------------------------------------------------------------ */

#ifndef NMATRIXOPS

    if (A->stype == 0)
    {
        /* (AA'+beta*I)x=b is the linear system that was solved */
        /* W = A'*X */
        W = cholmod_l_allocate_dense(A->ncol, 1, A->ncol, xtype, cm);
        cholmod_l_sdmult(A, 2, one, zero, X, W, cm);
        /* R = B - beta*X */
        cholmod_l_free_dense(&R, cm);
        R = cholmod_l_zeros(n, 1, xtype, cm);
        Rx = R->x;
        Xx = X->x;
        if (xtype == CHOLMOD_REAL)
        {
            for (int i = 0; i < n; i++)
            {
                Rx[i] = Bx[i] - beta[0] * Xx[i];
            }
        }
        /* R = A*W - R */
        cholmod_l_sdmult(A, 0, one, minusone, W, R, cm);
        cholmod_l_free_dense(&W, cm);
    }
    else
    {
        /* Ax=b was factorized and solved, R = B-A*X */
        cholmod_l_free_dense(&R, cm);
        R = cholmod_l_copy_dense(B, cm);
        cholmod_l_sdmult(A, 0, minusone, one, X, R, cm);
    }
    rnorm = -1;
    xnorm = 1;
    rnorm = cholmod_l_norm_dense(R, 0, cm);	    /* max abs. entry */
    xnorm = cholmod_l_norm_dense(X, 0, cm);	    /* max abs. entry */
    axbnorm = (anorm * xnorm + bnorm + ((n == 0) ? 1 : 0));
    resid = rnorm / axbnorm;
#else
    printf("residual not computed (requires CHOLMOD/MatrixOps)\n");
#endif
    /* ---------------------------------------------------------------------- */
    /* iterative refinement (real symmetric case only) */
    /* ---------------------------------------------------------------------- */

    resid2 = -1;
#ifndef NMATRIXOPS
    if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
    {
        cholmod_dense* R2;

        /* R2 = A\(B-A*X) */
        R2 = cholmod_l_solve(CHOLMOD_A, L, R, cm);
        /* compute X = X + A\(B-A*X) */
        Xx = X->x;
        Rx = R2->x;
        for (int i = 0; i < n; i++)
        {
            Xx[i] = Xx[i] + Rx[i];
        }
        cholmod_l_free_dense(&R2, cm);
        cholmod_l_free_dense(&R, cm);

        /* compute the new residual, R = B-A*X */
        cholmod_l_free_dense(&R, cm);
        R = cholmod_l_copy_dense(B, cm);
        cholmod_l_sdmult(A, 0, minusone, one, X, R, cm);
        rnorm2 = cholmod_l_norm_dense(R, 0, cm);
        resid2 = rnorm2 / axbnorm;
    }
#endif

    cholmod_l_free_dense(&R, cm);

    /* ---------------------------------------------------------------------- */
    /* print results */
    /* ---------------------------------------------------------------------- */

    anz = cm->anz;
    if(PRINT_DEFAULT){
        printf("ints in L: %15.0f, doubles in L: %15.0f\n",
            (double)isize, (double)xsize);
        printf("factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
            cm->fl, cm->lnz);
        if (A->stype == 0)
        {
            printf("nnz(A):    %15.0f\n", cm->anz);
        }
        else
        {
            printf("nnz(A*A'): %15.0f\n", cm->anz);
        }
        if (cm->lnz > 0)
        {
            printf("flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz);
        }
        if (anz > 0)
        {
            printf("nnz(L) / nnz(A): %8.1f\n", cm->lnz / cm->anz);
        }
        printf("analyze cputime:  %12.4f\n", ta);
        printf("factor  cputime:   %12.4f mflop: %8.1f\n", tf,
            (tf == 0) ? 0 : (1e-6 * cm->fl / tf));
        printf("solve   cputime:   %12.4f mflop: %8.1f\n", ts,
            (ts == 0) ? 0 : (1e-6 * 4 * cm->lnz / ts));
        printf("overall cputime:   %12.4f mflop: %8.1f\n",
            tot, (tot == 0) ? 0 : (1e-6 * (cm->fl + 4 * cm->lnz) / tot));
        printf("peak memory usage: %12.0f (MB)\n",
            (double)(cm->memory_usage) / 1048576.);
        printf("residual (|Ax-b|/(|A||x|+|b|)): %8.2e ", resid);

        printf("rcond    %8.1e\n\n", rcond);

        if (L_is_super)
        {
            cholmod_l_gpu_stats(cm);
        }
    }
    if (PRINT_TIME) {
        elapsedTime = stop - start;
        printf("all time CHOLMOD: analyze + factorize + solve = %10.6f sec\n", elapsedTime);
    }
    cholmod_l_free_factor(&L, cm);
    cholmod_l_free_dense(&X, cm);

    /* ---------------------------------------------------------------------- */
    /* free matrices and finish CHOLMOD */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free_sparse(&A, cm);
    cholmod_l_free_dense(&B, cm);
    cholmod_l_finish(cm);
}





int main() {
    FILE* inA, * inB;
    inA = fopen("../input/fidapm11.mtx", "r");
    inB = fopen("../input/B.vec", "r");
    if (inA == NULL || inB == NULL) {
        printf("Can't read input files");
        return -1;
    }
    int ver[3];
    printf("\n---------------------------------- cholmod_demo:\n");
    SuiteSparse_version(ver);
    printf("SuiteSparse version %d.%d.%d\n", ver[0], ver[1], ver[2]);
    printf("\n---------------------------------- cholmod_test_1:\n");
    cholmod_test(inA, inB, 0, ENABLE_CUDA, 1, 1);
    printf("\n---------------------------------- cholmod_test_2:\n");
    rewind(inA);
    rewind(inB);
    cholmod_test(inA, inB,0, DISABLE_CUDA, 1, 1);
    printf("\n---------------------------------- cholmod_test_3:\n");
    //rewind(inA);
    //rewind(inB);
    //cholmod_test_3(inA, inB);
    return 0;
}

//int main_test()
//{
//    double resid[4], t, ta, tf, ts[3], tot, bnorm, xnorm, anorm, rnorm, fl,
//        anz, axbnorm, rnorm2, resid2, rcond;
//    FILE* aFile, * bFile;
//    cholmod_sparse* A;
//    cholmod_dense* X = NULL, * B, * W, * R = NULL;
//    double one[2], zero[2], minusone[2], beta[2], xlnz;
//    cholmod_factor* L;
//    double* Bx, * Rx, * Xx;
//    int i, n, isize, xsize, ordering, xtype, s, ss, lnz;
//    int trial, method, L_is_super;
//    int ver[3];
//    aFile = fopen("../input/A.tri", "r");
//    bFile = fopen("../input/B.vec", "r");
//    ts[0] = 0.;
//    ts[1] = 0.;
//    ts[2] = 0.;
//
//    /* ---------------------------------------------------------------------- */
//    /* start CHOLMOD and set parameters */
//    /* ---------------------------------------------------------------------- */
//    cholmod_common* cm = (cholmod_common*)malloc(sizeof(cholmod_common));
//    cholmod_start(cm);
//    cm->useGPU = 1;
//
//    /* ---------------------------------------------------------------------- */
//    /* create basic scalars */
//    /* ---------------------------------------------------------------------- */
//
//    zero[0] = 0;
//    zero[1] = 0;
//    one[0] = 1;
//    one[1] = 0;
//    minusone[0] = -1;
//    minusone[1] = 0;
//    beta[0] = 1e-6;
//    beta[1] = 0;
//
//    /* ---------------------------------------------------------------------- */
//    /* read in a matrix */
//    /* ---------------------------------------------------------------------- */
//
//    printf("\n---------------------------------- cholmod_demo:\n");
//    SuiteSparse_version(ver);
//    printf("SuiteSparse version %d.%d.%d\n", ver[0], ver[1], ver[2]);
//    A = cholmod_read_sparse(aFile, cm);
//    xtype = A->xtype;
//    anorm = 1;
//    n = A->nrow;
//#ifndef NMATRIXOPS
//    anorm = cholmod_norm_sparse(A, 0, cm);
//    printf("norm (A,inf) = %g\n", anorm);
//    printf("norm (A,1)   = %g\n", cholmod_norm_sparse(A, 1, cm));
//#endif
//    cholmod_print_sparse(A, "A", cm);
//
//    if (A->nrow > A->ncol)
//    {
//        /* Transpose A so that A'A+beta*I will be factorized instead */
//        cholmod_sparse* C = cholmod_transpose(A, 2, cm);
//        cholmod_free_sparse(&A, cm);
//        A = C;
//        printf("transposing input matrix\n");
//    }
//    B = cholmod_read_dense(bFile, cm);
//    cholmod_print_dense(B, "B", cm);
//    Bx = B->x;
//    bnorm = 1;
//#ifndef nmatrixops
//    bnorm = cholmod_norm_dense(B, 0, cm);	/* max norm */
//    printf("bnorm %g\n", bnorm);
//#endif
//
//    /* ---------------------------------------------------------------------- */
//    /* analyze and factorize */
//    /* ---------------------------------------------------------------------- */
//
//    t = CPUTIME;
//    L = cholmod_analyze(A, cm);
//    ta = CPUTIME - t;
//    ta = MAX(ta, 0);
//
//    printf("Analyze: flop %g lnz %g\n", cm->fl, cm->lnz);
//
//    if (A->stype == 0)
//    {
//        printf("Factorizing A*A'+beta*I\n");
//        t = CPUTIME;
//        cholmod_factorize_p(A, beta, NULL, 0, L, cm);
//        tf = CPUTIME - t;
//        tf = MAX(tf, 0);
//    }
//    else
//    {
//        printf("Factorizing A\n");
//        t = CPUTIME;
//        cholmod_factorize(A, L, cm);
//        tf = CPUTIME - t;
//        tf = MAX(tf, 0);
//    }
//
//    cholmod_print_factor(L, "L", cm);
//
//    /* determine the # of integers's and reals's in L.  See cholmod_free */
//    if (L->is_super)
//    {
//        s = L->nsuper + 1;
//        xsize = L->xsize;
//        ss = L->ssize;
//        isize = n	/* L->Perm */
//            + n	/* L->ColCount, nz in each column of 'pure' L */
//            + s	/* L->pi, column pointers for L->s */
//            + s	/* L->px, column pointers for L->x */
//            + s	/* L->super, starting column index of each supernode */
//            + ss;	/* L->s, the pattern of the supernodes */
//    }
//    else
//    {
//        /* this space can increase if you change parameters to their non-
//         * default values (cm->final_pack, for example). */
//        lnz = L->nzmax;
//        xsize = lnz;
//        isize =
//            n	/* L->Perm */
//            + n	/* L->ColCount, nz in each column of 'pure' L */
//            + n + 1	/* L->p, column pointers */
//            + lnz	/* L->i, integer row indices */
//            + n	/* L->nz, nz in each column of L */
//            + n + 2	/* L->next, link list */
//            + n + 2;	/* L->prev, link list */
//    }
//
//    /* solve with Bset will change L from simplicial to supernodal */
//    rcond = cholmod_rcond(L, cm);
//    L_is_super = L->is_super;
//
//    /* ---------------------------------------------------------------------- */
//    /* solve */
//    /* ---------------------------------------------------------------------- */
//
//    for (method = 0; method <= 3; method++)
//    {
//        double x = n;
//        resid[method] = -1;       /* not yet computed */
//
//        if (method == 0)
//        {
//            /* basic solve, just once */
//            t = CPUTIME;
//            X = cholmod_solve(CHOLMOD_A, L, B, cm);
//            ts[0] = CPUTIME - t;
//            ts[0] = MAX(ts[0], 0);
//        }
//        else if (method == 2)
//        {
//            /* solve with reused workspace */
//            cholmod_dense* Ywork = NULL, * Ework = NULL;
//            cholmod_free_dense(&X, cm);
//
//            t = CPUTIME;
//            cholmod_solve2(CHOLMOD_A, L, B, NULL, &X, NULL,
//                &Ywork, &Ework, cm);
//            cholmod_free_dense(&Ywork, cm);
//            cholmod_free_dense(&Ework, cm);
//            ts[2] = CPUTIME - t;
//            ts[2] = MAX(ts[2], 0);
//        }
//        else
//        {
//            /* solve with reused workspace and sparse Bset */
//            cholmod_dense* Ywork = NULL, * Ework = NULL;
//            cholmod_dense* X2 = NULL, * B2 = NULL;
//            cholmod_sparse* Bset, * Xset = NULL;
//            int* Bsetp, * Bseti, * Xsetp, * Xseti, xlen, j, k, * Lnz;
//            double* X1x, * X2x, * B2x, err;
//            FILE* timelog = fopen("timelog.m", "w");
//            if (timelog) fprintf(timelog, "results = [\n");
//
//            B2 = cholmod_zeros(n, 1, xtype, cm);
//            B2x = B2->x;
//
//            Bset = cholmod_allocate_sparse(n, 1, 1, FALSE, TRUE, 0,
//                CHOLMOD_PATTERN, cm);
//            Bsetp = Bset->p;
//            Bseti = Bset->i;
//            Bsetp[0] = 0;     /* nnz(B) is 1 (it can be anything) */
//            Bsetp[1] = 1;
//            resid[3] = 0;
//
//            for (i = 0; i < n; i++)
//            {
//                /* B (i) is nonzero, all other entries are ignored
//                   (implied to be zero) */
//                Bseti[0] = i;
//                if (xtype == CHOLMOD_REAL)
//                {
//                    B2x[i] = Bx[i];
//                }
//
//                /* first get the entire solution, to compare against */
//                cholmod_solve2(CHOLMOD_A, L, B2, NULL, &X, NULL,
//                    &Ywork, &Ework, cm);
//
//                /* now get the sparse solutions; this will change L from
//                   supernodal to simplicial */
//
//                if (i == 0)
//                {
//                    /* first solve can be slower because it has to allocate
//                       space for X2, Xset, etc, and change L.
//                       So don't time it */
//                    cholmod_solve2(CHOLMOD_A, L, B2, Bset, &X2, &Xset,
//                        &Ywork, &Ework, cm);
//                }
//
//                t = CPUTIME;
//                /* solve Ax=b but only to get x(i).
//                    b is all zero except for b(i).
//                    This takes O(xlen) time */
//                cholmod_solve2(CHOLMOD_A, L, B2, Bset, &X2, &Xset,
//                    &Ywork, &Ework, cm);
//                t = CPUTIME - t;
//                t = MAX(t, 0);
//
//                /* check the solution and log the time */
//                Xsetp = Xset->p;
//                Xseti = Xset->i;
//                xlen = Xsetp[1];
//                X1x = X->x;
//                X2x = X2->x;
//                Lnz = L->nz;
//
//                /*
//                printf ("\ni %d xlen %d  (%p %p)\n", i, xlen, X1x, X2x) ;
//                */
//
//                if (xtype == CHOLMOD_REAL)
//                {
//                    fl = 2 * xlen;
//                    for (k = 0; k < xlen; k++)
//                    {
//                        j = Xseti[k];
//                        fl += 4 * Lnz[j];
//                        err = X1x[j] - X2x[j];
//                        err = ABS(err);
//                        resid[3] = MAX(resid[3], err);
//                    }
//                }
//                if (timelog) fprintf(timelog, "%g %g %g %g\n",
//                    (double)i, (double)xlen, fl, t);
//
//                /* clear B for the next test */
//                if (xtype == CHOLMOD_REAL)
//                {
//                    B2x[i] = 0;
//                }
//            }
//
//            if (timelog)
//            {
//                fprintf(timelog, "] ; resid = %g ;\n", resid[3]);
//                fprintf(timelog, "lnz = %g ;\n", cm->lnz);
//                fprintf(timelog, "t = %g ;   %% dense solve time\n", ts[2]);
//                fclose(timelog);
//            }
//
//#ifndef NMATRIXOPS
//            resid[3] = resid[3] / cholmod_norm_dense(X, 1, cm);
//#endif
//
//            cholmod_free_dense(&Ywork, cm);
//            cholmod_free_dense(&Ework, cm);
//            cholmod_free_dense(&X2, cm);
//            cholmod_free_dense(&B2, cm);
//            cholmod_free_sparse(&Xset, cm);
//            cholmod_free_sparse(&Bset, cm);
//        }
//
//        /* ------------------------------------------------------------------ */
//        /* compute the residual */
//        /* ------------------------------------------------------------------ */
//
//        if (method < 3)
//        {
//#ifndef NMATRIXOPS
//
//            if (A->stype == 0)
//            {
//                /* (AA'+beta*I)x=b is the linear system that was solved */
//                /* W = A'*X */
//                W = cholmod_allocate_dense(A->ncol, 1, A->ncol, xtype, cm);
//                cholmod_sdmult(A, 2, one, zero, X, W, cm);
//                /* R = B - beta*X */
//                cholmod_free_dense(&R, cm);
//                R = cholmod_zeros(n, 1, xtype, cm);
//                Rx = R->x;
//                Xx = X->x;
//                if (xtype == CHOLMOD_REAL)
//                {
//                    for (i = 0; i < n; i++)
//                    {
//                        Rx[i] = Bx[i] - beta[0] * Xx[i];
//                    }
//                }
//                /* R = A*W - R */
//                cholmod_sdmult(A, 0, one, minusone, W, R, cm);
//                cholmod_free_dense(&W, cm);
//            }
//            else
//            {
//                /* Ax=b was factorized and solved, R = B-A*X */
//                cholmod_free_dense(&R, cm);
//                R = cholmod_copy_dense(B, cm);
//                cholmod_sdmult(A, 0, minusone, one, X, R, cm);
//            }
//            rnorm = -1;
//            xnorm = 1;
//            rnorm = cholmod_norm_dense(R, 0, cm);	    /* max abs. entry */
//            xnorm = cholmod_norm_dense(X, 0, cm);	    /* max abs. entry */
//            axbnorm = (anorm * xnorm + bnorm + ((n == 0) ? 1 : 0));
//            resid[method] = rnorm / axbnorm;
//#else
//            printf("residual not computed (requires CHOLMOD/MatrixOps)\n");
//#endif
//        }
//    }
//
//    tot = ta + tf + ts[0];
//
//    /* ---------------------------------------------------------------------- */
//    /* iterative refinement (real symmetric case only) */
//    /* ---------------------------------------------------------------------- */
//
//    resid2 = -1;
//#ifndef NMATRIXOPS
//    if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
//    {
//        cholmod_dense* R2;
//
//        /* R2 = A\(B-A*X) */
//        R2 = cholmod_solve(CHOLMOD_A, L, R, cm);
//        /* compute X = X + A\(B-A*X) */
//        Xx = X->x;
//        Rx = R2->x;
//        for (i = 0; i < n; i++)
//        {
//            Xx[i] = Xx[i] + Rx[i];
//        }
//        cholmod_free_dense(&R2, cm);
//        cholmod_free_dense(&R, cm);
//
//        /* compute the new residual, R = B-A*X */
//        cholmod_free_dense(&R, cm);
//        R = cholmod_copy_dense(B, cm);
//        cholmod_sdmult(A, 0, minusone, one, X, R, cm);
//        rnorm2 = cholmod_norm_dense(R, 0, cm);
//        resid2 = rnorm2 / axbnorm;
//    }
//#endif
//
//    cholmod_free_dense(&R, cm);
//
//    /* ---------------------------------------------------------------------- */
//    /* print results */
//    /* ---------------------------------------------------------------------- */
//
//    anz = cm->anz;
//    for (i = 0; i < CHOLMOD_MAXMETHODS; i++)
//    {
//        fl = cm->method[i].fl;
//        xlnz = cm->method[i].lnz;
//        cm->method[i].fl = -1;
//        cm->method[i].lnz = -1;
//        ordering = cm->method[i].ordering;
//        if (fl >= 0)
//        {
//            printf("Ordering: ");
//            if (ordering == CHOLMOD_POSTORDERED) printf("postordered ");
//            if (ordering == CHOLMOD_NATURAL)     printf("natural ");
//            if (ordering == CHOLMOD_GIVEN)	     printf("user    ");
//            if (ordering == CHOLMOD_AMD)	     printf("AMD     ");
//            if (ordering == CHOLMOD_METIS)	     printf("METIS   ");
//            if (ordering == CHOLMOD_NESDIS)      printf("NESDIS  ");
//            if (xlnz > 0)
//            {
//                printf("fl/lnz %10.1f", fl / xlnz);
//            }
//            if (anz > 0)
//            {
//                printf("  lnz/anz %10.1f", xlnz / anz);
//            }
//            printf("\n");
//        }
//    }
//
//    printf("ints in L: %15.0f, doubles in L: %15.0f\n",
//        (double)isize, (double)xsize);
//    printf("factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
//        cm->fl, cm->lnz);
//    if (A->stype == 0)
//    {
//        printf("nnz(A):    %15.0f\n", cm->anz);
//    }
//    else
//    {
//        printf("nnz(A*A'): %15.0f\n", cm->anz);
//    }
//    if (cm->lnz > 0)
//    {
//        printf("flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz);
//    }
//    if (anz > 0)
//    {
//        printf("nnz(L) / nnz(A): %8.1f\n", cm->lnz / cm->anz);
//    }
//    printf("analyze cputime:  %12.4f\n", ta);
//    printf("factor  cputime:   %12.4f mflop: %8.1f\n", tf,
//        (tf == 0) ? 0 : (1e-6 * cm->fl / tf));
//    printf("solve   cputime:   %12.4f mflop: %8.1f\n", ts[0],
//        (ts[0] == 0) ? 0 : (1e-6 * 4 * cm->lnz / ts[0]));
//    printf("overall cputime:   %12.4f mflop: %8.1f\n",
//        tot, (tot == 0) ? 0 : (1e-6 * (cm->fl + 4 * cm->lnz) / tot));
//    printf("solve   cputime:   %12.4f mflop: %8.1f (%d trials)\n", ts[1],
//        (ts[1] == 0) ? 0 : (1e-6 * 4 * cm->lnz / ts[1]), NTRIALS);
//    printf("solve2  cputime:   %12.4f mflop: %8.1f (%d trials)\n", ts[2],
//        (ts[2] == 0) ? 0 : (1e-6 * 4 * cm->lnz / ts[2]), NTRIALS);
//    printf("peak memory usage: %12.0f (MB)\n",
//        (double)(cm->memory_usage) / 1048576.);
//    printf("residual (|Ax-b|/(|A||x|+|b|)): ");
//    for (method = 0; method <= 3; method++)
//    {
//        printf("%8.2e ", resid[method]);
//    }
//    printf("\n");
//    if (resid2 >= 0)
//    {
//        printf("residual %8.1e (|Ax-b|/(|A||x|+|b|))"
//            " after iterative refinement\n", resid2);
//    }
//
//    printf("rcond    %8.1e\n\n", rcond);
//
//    if (L_is_super)
//    {
//        cholmod_gpu_stats(cm);
//    }
//
//    cholmod_free_factor(&L, cm);
//    cholmod_free_dense(&X, cm);
//
//    /* ---------------------------------------------------------------------- */
//    /* free matrices and finish CHOLMOD */
//    /* ---------------------------------------------------------------------- */
//
//    cholmod_free_sparse(&A, cm);
//    cholmod_free_dense(&B, cm);
//    cholmod_finish(cm);
//
//    return (0);
//}
