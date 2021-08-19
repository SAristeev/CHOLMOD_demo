#include "cholmod.h"
#include "CHOLMOD_demo.h"

void cholmod_test(FILE* inA, FILE* inB, FILE * outX, FILE* FID, int SOLVER_VER, FILE * log) {
    double startTime, analyzeTime, factorizeTime, solveTime, stopTime;
    cholmod_sparse* A;
    cholmod_dense* X = NULL, * B = NULL, * W, * R = NULL, * fid_X = NULL;
    cholmod_factor* L;
    double* Bx, * Xx, * Rx, *Fx;
    double resid, resid2, t, ta, tf, ts, tot, anorm, bnorm, rcond, anz, xnorm, rnorm, rnorm2,
        axbnorm;
    double one[2], zero[2], minusone[2], beta[2];
    int n, isize, xsize, xtype, s, ss, lnz;
    int L_is_super;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */
    /* ---------------------------------------------------------------------- */

    cholmod_common* cm = (cholmod_common*)malloc(sizeof(cholmod_common));
    cholmod_l_start(cm);
    cm->useGPU = 1;

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
    anorm = cholmod_l_norm_sparse(A, 0, cm);
  
    fprintf(log, "norm (A,inf)             --- %g\n", anorm);
    fprintf(log, "norm (A,1)               --- %g\n", cholmod_l_norm_sparse(A, 1, cm));

    cholmod_l_print_sparse(A, "A", cm);

    /* ---------------------------------------------------------------------- */
    /* read vector B */
    /* ---------------------------------------------------------------------- */

    //B = cholmod_l_ones(A->nrow, 1.0, A->xtype, cm);   /* b = ones(n,1) */
    B = cholmod_l_read_dense(inB, cm);
    cholmod_l_print_dense(B, "B", cm);
    Bx = B->x;
    bnorm = cholmod_l_norm_dense(B, 0, cm);	/* max norm */
    fprintf(log, "bnorm                    --- %g\n", bnorm);

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    startTime = second();
    L = cholmod_l_analyze(A, cm);
    analyzeTime = second();

    if (A->stype == 0)
        cholmod_l_factorize_p(A, beta, NULL, 0, L, cm);
    else
        cholmod_l_factorize(A, L, cm);
    factorizeTime = second();

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

    solveTime = second();
    
    if (SOLVER_VER == 0) {
        /* basic solve */
        X = cholmod_l_solve(CHOLMOD_A, L, B, cm);
    }

    else {
        /* solve with reused workspace */
        cholmod_dense* Ywork = NULL, * Ework = NULL;
        cholmod_l_solve2(CHOLMOD_A, L, B, NULL, &X, NULL,
            &Ywork, &Ework, cm);
        cholmod_l_free_dense(&Ywork, cm);
        cholmod_l_free_dense(&Ework, cm);
    }

    stopTime = second();

    fprintf(log, "bnorm                    --- %g\n", bnorm);

    Xx = X->x;
    rewind(FID);
    fid_X = cholmod_l_read_dense(FID, cm);
    Fx = fid_X->x;
    for (int i = 0; i < n; i++)
    {
        Fx[i] = Fx[i] - Xx[i];
    }
    
    fprintf(log, "|Fid(X)-X|               --- %8.2e\n", cholmod_l_norm_dense(fid_X, 0, cm));

    /* ------------------------------------------------------------------ */
    /* compute the residual */
    /* ------------------------------------------------------------------ */

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

    /* ---------------------------------------------------------------------- */
    /* iterative refinement (real symmetric case only) */
    /* ---------------------------------------------------------------------- */

    resid2 = -1;
    if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
    {
        cholmod_dense* R2;

        /* R2 = A\(B-A*X) */
        R2 = cholmod_l_solve(CHOLMOD_A, L, R, cm);
        /* compute X = X + A\(B-A*X) */
        
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

    cholmod_l_write_dense(outX, X, NULL, cm);
    cholmod_l_free_dense(&R, cm);

    /* ---------------------------------------------------------------------- */
    /* print results */
    /* ---------------------------------------------------------------------- */

    anz = cm->anz;
    

    fprintf(log, "Analyze   time           --- %8.6f seconds\n", analyzeTime - startTime);
    fprintf(log, "Factorize time           --- %8.6f seconds\n", factorizeTime - analyzeTime);
    fprintf(log, "Solve     time           --- %8.6f seconds\n", stopTime - solveTime);
    fprintf(log, "All       time           --- %8.6f seconds\n", stopTime - startTime);
    
    if (A->stype == 0)
        fprintf(log, "nnz(A):                  --- %10.6f\n", cm->anz);
    else
    {
        fprintf(log, "nnz(A*A'):               --- %10.6f\n", cm->anz);
    }
    fprintf(log, "peak memory usage:           %12.0f (MB)\n", (double)(cm->memory_usage) / 1048576.);
    fprintf(log, "residual |Ax-b|/(|A||x|+|b|):%8.2e\n ", resid);

    if (L_is_super)
    {
        cholmod_l_gpu_stats(cm);
    }

    /* ---------------------------------------------------------------------- */
    /* free matrices and finish CHOLMOD */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free_factor(&L, cm);
    cholmod_l_free_dense(&X, cm);
    cholmod_l_free_dense(&fid_X, cm);
    cholmod_l_free_sparse(&A, cm);
    cholmod_l_free_dense(&B, cm);
    cholmod_l_finish(cm);

    rewind(inA);
    rewind(inB);
}




int main() {
    FILE* inA, * inB, * outX1, *outX2,*FID,*log;
    inA = fopen("../input/A.tri", "r");
    inB = fopen("../input/B.vec", "r");
    outX1 = fopen("../output/X1.vec", "w");
    outX2 = fopen("../output/X2.vec", "w");
    FID = fopen("../input/X.vec", "r");
    log = fopen("../log/CHOLMOD.log", "w");

    if (inA == NULL || inB == NULL || FID == NULL) {
        printf("Can't read input files");
        return -1;
    }
    int ver[3];
    SuiteSparse_version(ver);
    fprintf(log, "SuiteSparse version %d.%d.%d\n", ver[0], ver[1], ver[2]);

    fprintf(log, "\nTest 1: Simple Solver\n");
    cholmod_test(inA, inB, outX1, FID ,0, log);

    fprintf(log, "\nTest 2: Solver with reused workspace\n");
    cholmod_test(inA, inB, outX2, FID, 1, log);
    
    printf("\nDone\nSee info on 'log/CHOLMOD.log'\n");

    fclose(inA);
    fclose(inB);
    fclose(outX1);
    fclose(outX2);
    fclose(log);
    fclose(FID);

    inA = NULL;
    inB = NULL;
    outX1 = NULL;
    outX2 = NULL;
    log = NULL;
    FID = NULL;

    return 0;
}