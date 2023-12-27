/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
   Interface:    Linear-Algebraic (IJ)

   Compile with: make custom_amg_cycle

   Sample run:   mpirun -np 4 custom_amg_cycle

   Description:  This example solves the 2-D Laplacian problem with zero boundary
                 conditions on an n x n grid.  The number of unknowns is N=n^2.
                 The standard 5-point stencil is used, and we solve for the
                 interior nodes only.

                 This code solves the same problem as Example 3 and 5.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "ex.h"


#include "_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "../parcsr_block_mv/par_csr_block_matrix.h"

HYPRE_Int compute_residual(hypre_ParAMGData *amg_data, HYPRE_Int level, hypre_ParCSRMatrix **A_array, hypre_ParVector **F_array, hypre_ParVector **U_array, hypre_ParVector *Vtemp) {
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Residual");
   HYPRE_Real alpha = -1.0;
   HYPRE_Real beta = 1.0;
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level], U_array[level],
                                      beta, F_array[level], Vtemp);
   HYPRE_ANNOTATE_REGION_END("%s", "Residual");
   return level;
}

HYPRE_Int apply_restriction(hypre_ParAMGData *amg_data, HYPRE_Int level, hypre_ParCSRMatrix **A_array, hypre_ParVector **F_array, hypre_ParVector **U_array, hypre_ParVector *Vtemp, HYPRE_Int restri_type, hypre_ParCSRMatrix **R_array) {
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Restriction");
   HYPRE_Int new_level = level+1;
   hypre_ParVectorSetZeros(U_array[new_level]);
   HYPRE_Real alpha = 1.0;
   HYPRE_Real beta = 0.0;
   if (restri_type)
   {
      /* RL: no transpose for R */
      hypre_ParCSRMatrixMatvec(alpha, R_array[level], Vtemp,
                               beta, F_array[new_level]);
   }
   else
   {
      hypre_ParCSRMatrixMatvecT(alpha, R_array[level], Vtemp,
                                beta, F_array[new_level]);
   }
   HYPRE_ANNOTATE_REGION_END("%s", "Restriction");
   return new_level;

}

HYPRE_Int apply_prolongation(hypre_ParAMGData *amg_data, HYPRE_Int level, hypre_ParCSRMatrix **A_array, hypre_ParVector **F_array, hypre_ParVector **U_array, hypre_ParVector *Vtemp, hypre_ParCSRMatrix **P_array) {
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");
   HYPRE_Real alpha = 1.0;
   HYPRE_Real beta = 1.0;
   HYPRE_Int new_level = level-1;
   hypre_ParCSRMatrixMatvec(alpha, P_array[new_level],
                            U_array[level],
                            beta, U_array[new_level]);
   hypre_ParVectorAllZeros(U_array[new_level]) = 0;
   HYPRE_Int local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[new_level]));
   hypre_ParVectorSetLocalSize(Vtemp, local_size);
   HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");
   return new_level;
}

HYPRE_Int apply_coarse_solver(hypre_ParAMGData *amg_data, HYPRE_Int level, hypre_ParCSRMatrix **A_array, hypre_ParVector **F_array, hypre_ParVector **U_array, hypre_ParVector *Vtemp) {
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarse Solve");
   HYPRE_Int relax_type = 9;
   hypre_GaussElimSetup(amg_data, level, relax_type);
   hypre_GaussElimSolve(amg_data, level, relax_type);
   HYPRE_ANNOTATE_REGION_END("%s", "Coarse Solve");
   return level;
}

HYPRE_Int apply_smoother(hypre_ParAMGData *amg_data, HYPRE_Int level, hypre_ParCSRMatrix **A_array, hypre_ParVector **F_array, hypre_ParVector **U_array, 
hypre_ParVector *Vtemp, hypre_ParVector *Ztemp, HYPRE_Int *CF_marker, HYPRE_Int relax_local, HYPRE_Int cycle_param, 
HYPRE_Real relax_weight_level, HYPRE_Real omega_level, hypre_Vector *l1_norms_level) {
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
   HYPRE_Int relax_type = 3;
   HYPRE_Int Solve_err_flag = hypre_BoomerAMGRelaxIF(A_array[level],
                                           F_array[level],
                                           CF_marker,
                                           relax_type,
                                           relax_local,
                                           cycle_param,
                                           relax_weight_level,
                                           omega_level,
                                           l1_norms_level ? hypre_VectorData(l1_norms_level) : NULL,
                                           U_array[level],
                                           Vtemp,
                                           Ztemp);
   HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
   return level;
}

HYPRE_Int
BoomerAMGCycle( void              *amg_vdata,
                      hypre_ParVector  **F_array,
                      hypre_ParVector  **U_array   )
{
   hypre_ParAMGData *amg_data = (hypre_ParAMGData*) amg_vdata;

   HYPRE_Solver *smoother;

   /* Data Structure variables */
   hypre_ParCSRMatrix      **A_array;
   hypre_ParCSRMatrix      **P_array;
   hypre_ParCSRMatrix      **R_array;
   hypre_ParVector          *Utemp = NULL;
   hypre_ParVector          *Vtemp;
   hypre_ParVector          *Rtemp;
   hypre_ParVector          *Ptemp;
   hypre_ParVector          *Ztemp;
   hypre_ParVector          *Aux_U;
   hypre_ParVector          *Aux_F;
   hypre_ParCSRBlockMatrix **A_block_array;
   hypre_ParCSRBlockMatrix **P_block_array;
   hypre_ParCSRBlockMatrix **R_block_array;

   HYPRE_Real      *Ztemp_data = NULL;
   HYPRE_Real      *Ptemp_data = NULL;
   hypre_IntArray **CF_marker_array;
   HYPRE_Int       *CF_marker;
   /*
   HYPRE_Int     **unknown_map_array;
   HYPRE_Int     **point_map_array;
   HYPRE_Int     **v_at_point_array;
   */
   HYPRE_Real      cycle_op_count;
   HYPRE_Int       cycle_type;
   HYPRE_Int       fcycle, fcycle_lev;
   HYPRE_Int       num_levels;
   HYPRE_Int       max_levels;
   HYPRE_Real     *num_coeffs;
   HYPRE_Int      *num_grid_sweeps;
   HYPRE_Int      *grid_relax_type;
   HYPRE_Int     **grid_relax_points;
   HYPRE_Int       block_mode;
   HYPRE_Int       cheby_order;

   /* Local variables  */
   HYPRE_Int      *lev_counter;
   HYPRE_Int       Solve_err_flag;
   HYPRE_Int       k;
   HYPRE_Int       i, j, jj;
   HYPRE_Int       level;
   HYPRE_Int       cycle_param;
   HYPRE_Int       coarse_grid;
   HYPRE_Int       fine_grid;
   HYPRE_Int       Not_Finished;
   HYPRE_Int       num_sweep;
   HYPRE_Int       cg_num_sweep = 1;
   HYPRE_Int       relax_type;
   HYPRE_Int       relax_points = 0;
   HYPRE_Int       relax_order;
   HYPRE_Int       relax_local;
   HYPRE_Int       old_version = 0;
   HYPRE_Real     *relax_weight;
   HYPRE_Real     *omega;
   HYPRE_Real      alfa, beta, gammaold;
   HYPRE_Real      gamma = 1.0;
   HYPRE_Int       local_size = 0;
   /*   HYPRE_Int      *smooth_option; */
   HYPRE_Int       smooth_type;
   HYPRE_Int       smooth_num_levels;
   HYPRE_Int       my_id;
   HYPRE_Int       restri_type;
   HYPRE_Real      alpha;
   hypre_Vector  **l1_norms = NULL;
   hypre_Vector   *l1_norms_level;
   hypre_Vector  **ds = hypre_ParAMGDataChebyDS(amg_data);
   HYPRE_Real    **coefs = hypre_ParAMGDataChebyCoefs(amg_data);
   HYPRE_Int       seq_cg = 0;
   HYPRE_Int       partial_cycle_coarsest_level;
   HYPRE_Int       partial_cycle_control;
   MPI_Comm        comm;

   char            nvtx_name[1024];

#if 0
   HYPRE_Real   *D_mat;
   HYPRE_Real   *S_vec;
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Acquire data and allocate storage */
   A_array           = hypre_ParAMGDataAArray(amg_data);
   P_array           = hypre_ParAMGDataPArray(amg_data);
   R_array           = hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = hypre_ParAMGDataVtemp(amg_data);
   Rtemp             = hypre_ParAMGDataRtemp(amg_data);
   Ptemp             = hypre_ParAMGDataPtemp(amg_data);
   Ztemp             = hypre_ParAMGDataZtemp(amg_data);
   num_levels        = hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = hypre_ParAMGDataCycleType(amg_data);
   fcycle            = hypre_ParAMGDataFCycle(amg_data);

   A_block_array     = hypre_ParAMGDataABlockArray(amg_data);
   P_block_array     = hypre_ParAMGDataPBlockArray(amg_data);
   R_block_array     = hypre_ParAMGDataRBlockArray(amg_data);
   block_mode        = hypre_ParAMGDataBlockMode(amg_data);

   num_grid_sweeps     = hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = hypre_ParAMGDataRelaxWeight(amg_data);
   omega               = hypre_ParAMGDataOmega(amg_data);
   smooth_type         = hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels   = hypre_ParAMGDataSmoothNumLevels(amg_data);
   l1_norms            = hypre_ParAMGDataL1Norms(amg_data);
   /* smooth_option       = hypre_ParAMGDataSmoothOption(amg_data); */
   /* RL */
   restri_type = hypre_ParAMGDataRestriction(amg_data);

   partial_cycle_coarsest_level = hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data);
   partial_cycle_control = hypre_ParAMGDataPartialCycleControl(amg_data);

   /*max_eig_est = hypre_ParAMGDataMaxEigEst(amg_data);
   min_eig_est = hypre_ParAMGDataMinEigEst(amg_data);
   cheby_fraction = hypre_ParAMGDataChebyFraction(amg_data);*/
   cheby_order = hypre_ParAMGDataChebyOrder(amg_data);

   cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = hypre_CTAlloc(HYPRE_Int, num_levels, HYPRE_MEMORY_HOST);

   if (hypre_ParAMGDataParticipate(amg_data))
   {
      seq_cg = 1;
   }

   /* Initialize */
   Solve_err_flag = 0;

   num_coeffs = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
   num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = hypre_ParCSRMatrixComm(A_array[0]);
   hypre_MPI_Comm_rank(comm, &my_id);

   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j] = hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
   }

   level = 0;
   cycle_param = 1;
   relax_points = 0;
   relax_local = 0;
   l1_norms_level = NULL;

   local_size = hypre_VectorSize(hypre_ParVectorLocalVector(F_array[level]));
   hypre_ParVectorSetLocalSize(Vtemp, local_size);
   //hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
   level = apply_smoother(amg_data, level, A_array, F_array, U_array, Vtemp, Ztemp, CF_marker, relax_local, 
                          cycle_param, relax_weight[level], omega[level], l1_norms_level); 

   level = compute_residual(amg_data, level, A_array, F_array, U_array, Vtemp); 


   level = apply_restriction(amg_data, level, A_array, F_array, U_array, Vtemp, restri_type, R_array); 

   level = apply_coarse_solver(amg_data, level, A_array, F_array, U_array, Vtemp);  

   level = apply_prolongation(amg_data, level, A_array, F_array, U_array, Vtemp, P_array); 

   level = apply_smoother(amg_data, level, A_array, F_array, U_array, Vtemp, Ztemp, CF_marker, relax_local, 
                          cycle_param, relax_weight[level], omega[level], l1_norms_level); 

   HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}

HYPRE_Int
BoomerAMGSolve( void               *amg_vdata,
                      hypre_ParCSRMatrix *A,
                      hypre_ParVector    *f,
                      hypre_ParVector    *u         )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);
   hypre_ParAMGData    *amg_data = (hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   HYPRE_Int            amg_print_level;
   HYPRE_Int            amg_logging;
   HYPRE_Int            cycle_count;
   HYPRE_Int            num_levels;
   HYPRE_Int            converge_type;
   HYPRE_Int            block_mode;
   HYPRE_Int            additive;
   HYPRE_Int            mult_additive;
   HYPRE_Int            simple;
   HYPRE_Int            min_iter;
   HYPRE_Int            max_iter;
   HYPRE_Real           tol;

   hypre_ParCSRMatrix **A_array;
   hypre_ParVector    **F_array;
   hypre_ParVector    **U_array;

   hypre_ParCSRBlockMatrix **A_block_array;

   /*  Local variables  */
   HYPRE_Int           j;
   HYPRE_Int           Solve_err_flag;
   HYPRE_Int           num_procs, my_id;
   HYPRE_Int           num_vectors;
   HYPRE_Real          alpha = 1.0;
   HYPRE_Real          beta = -1.0;
   HYPRE_Real          cycle_op_count;
   HYPRE_Real          total_coeffs;
   HYPRE_Real          total_variables;
   HYPRE_Real         *num_coeffs;
   HYPRE_Real         *num_variables;
   HYPRE_Real          cycle_cmplxty = 0.0;
   HYPRE_Real          operat_cmplxty;
   HYPRE_Real          grid_cmplxty;
   HYPRE_Real          conv_factor = 0.0;
   HYPRE_Real          resid_nrm = 1.0;
   HYPRE_Real          resid_nrm_init = 0.0;
   HYPRE_Real          relative_resid;
   HYPRE_Real          rhs_norm = 0.0;
   HYPRE_Real          old_resid;
   HYPRE_Real          ieee_check = 0.;

   hypre_ParVector    *Vtemp;
   hypre_ParVector    *Rtemp;
   hypre_ParVector    *Ptemp;
   hypre_ParVector    *Ztemp;
   hypre_ParVector    *Residual = NULL;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   amg_print_level  = hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging      = hypre_ParAMGDataLogging(amg_data);
   if (amg_logging > 1)
   {
      Residual = hypre_ParAMGDataResidual(amg_data);
   }
   num_levels       = hypre_ParAMGDataNumLevels(amg_data);
   A_array          = hypre_ParAMGDataAArray(amg_data);
   F_array          = hypre_ParAMGDataFArray(amg_data);
   U_array          = hypre_ParAMGDataUArray(amg_data);

   converge_type    = hypre_ParAMGDataConvergeType(amg_data);
   tol              = hypre_ParAMGDataTol(amg_data);
   min_iter         = hypre_ParAMGDataMinIter(amg_data);
   max_iter         = hypre_ParAMGDataMaxIter(amg_data);
   additive         = hypre_ParAMGDataAdditive(amg_data);
   simple           = hypre_ParAMGDataSimple(amg_data);
   mult_additive    = hypre_ParAMGDataMultAdditive(amg_data);
   block_mode       = hypre_ParAMGDataBlockMode(amg_data);
   A_block_array    = hypre_ParAMGDataABlockArray(amg_data);
   Vtemp            = hypre_ParAMGDataVtemp(amg_data);
   Rtemp            = hypre_ParAMGDataRtemp(amg_data);
   Ptemp            = hypre_ParAMGDataPtemp(amg_data);
   Ztemp            = hypre_ParAMGDataZtemp(amg_data);
   num_vectors      = hypre_ParVectorNumVectors(f);

   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   /* Verify that the number of vectors held by f and u match */
   if (hypre_ParVectorNumVectors(f) !=
       hypre_ParVectorNumVectors(u))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: num_vectors for RHS and LHS do not match!\n");
      return hypre_error_flag;
   }

   /* Update work vectors */
   hypre_ParVectorResize(Vtemp, num_vectors);
   hypre_ParVectorResize(Rtemp, num_vectors);
   hypre_ParVectorResize(Ptemp, num_vectors);
   hypre_ParVectorResize(Ztemp, num_vectors);
   if (amg_logging > 1)
   {
      hypre_ParVectorResize(Residual, num_vectors);
   }
   for (j = 1; j < num_levels; j++)
   {
      hypre_ParVectorResize(F_array[j], num_vectors);
      hypre_ParVectorResize(U_array[j], num_vectors);
   }

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1)
   {
      hypre_BoomerAMGWriteSolverParams(amg_data);
   }

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   cycle_count = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1 && tol > 0.)
   {
      hypre_printf("\n\nAMG SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print
    *-----------------------------------------------------------------------*/

   if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
   {
      if ( amg_logging > 1 )
      {
         hypre_ParVectorCopy(F_array[0], Residual);
         if (tol > 0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Residual);
         }
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd( Residual, Residual ));
      }
      else
      {
         hypre_ParVectorCopy(F_array[0], Vtemp);
         if (tol > 0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
         }
         resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resid_nrm != 0.)
      {
         ieee_check = resid_nrm / resid_nrm; /* INF -> NaN conversion */
      }

      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (amg_print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_BoomerAMGSolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }

      /* r0 */
      resid_nrm_init = resid_nrm;

      if (0 == converge_type)
      {
         rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
         if (rhs_norm)
         {
            relative_resid = resid_nrm_init / rhs_norm;
         }
         else
         {
            relative_resid = resid_nrm_init;
         }
      }
      else
      {
         /* converge_type != 0, test convergence with ||r|| / ||r0|| */
         relative_resid = 1.0;
      }
   }
   else
   {
      relative_resid = 1.;
   }

   if (my_id == 0 && amg_print_level > 1)
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", resid_nrm_init,
                   relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/

   while ( (relative_resid >= tol || cycle_count < min_iter) && cycle_count < max_iter )
   {
      hypre_ParAMGDataCycleOpCount(amg_data) = 0;

      BoomerAMGCycle(amg_data, F_array, U_array);
      

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if (amg_print_level > 1 || amg_logging > 1 || tol > 0.)
      {
         old_resid = resid_nrm;

         if (amg_logging > 1)
         {
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta, F_array[0],
                                               Residual);
            resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Residual, Residual));
         }
         else
         {
            hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[0], U_array[0], beta, F_array[0],
                                               Vtemp);
            resid_nrm = hypre_sqrt(hypre_ParVectorInnerProd(Vtemp, Vtemp));
         }

         if (old_resid)
         {
            conv_factor = resid_nrm / old_resid;
         }
         else
         {
            conv_factor = resid_nrm;
         }

         if (0 == converge_type)
         {
            if (rhs_norm)
            {
               relative_resid = resid_nrm / rhs_norm;
            }
            else
            {
               relative_resid = resid_nrm;
            }
         }
         else
         {
            relative_resid = resid_nrm / resid_nrm_init;
         }

         hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      }

      ++cycle_count;

      hypre_ParAMGDataNumIterations(amg_data) = cycle_count;
#ifdef CUMNUMIT
      ++hypre_ParAMGDataCumNumIterations(amg_data);
#endif

      if (my_id == 0 && amg_print_level > 1)
      {
         hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      hypre_error(HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   if (cycle_count > 0 && resid_nrm_init)
   {
      conv_factor = hypre_pow((resid_nrm / resid_nrm_init), (1.0 / (HYPRE_Real) cycle_count));
   }
   else
   {
      conv_factor = 1.;
   }

   if (amg_print_level > 1)
   {
      num_coeffs       = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
      num_variables    = hypre_CTAlloc(HYPRE_Real,  num_levels, HYPRE_MEMORY_HOST);
      num_coeffs[0]    = hypre_ParCSRMatrixDNumNonzeros(A);
      num_variables[0] = hypre_ParCSRMatrixGlobalNumRows(A);

      if (block_mode)
      {
         for (j = 1; j < num_levels; j++)
         {
            num_coeffs[j]    = (HYPRE_Real) hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);
            num_variables[j] = (HYPRE_Real) hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[j]);
         }
         num_coeffs[0]    = hypre_ParCSRBlockMatrixDNumNonzeros(A_block_array[0]);
         num_variables[0] = hypre_ParCSRBlockMatrixGlobalNumRows(A_block_array[0]);

      }
      else
      {
         for (j = 1; j < num_levels; j++)
         {
            num_coeffs[j]    = (HYPRE_Real) hypre_ParCSRMatrixNumNonzeros(A_array[j]);
            num_variables[j] = (HYPRE_Real) hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
         }
      }


      for (j = 0; j < hypre_ParAMGDataNumLevels(amg_data); j++)
      {
         total_coeffs += num_coeffs[j];
         total_variables += num_variables[j];
      }

      cycle_op_count = hypre_ParAMGDataCycleOpCount(amg_data);

      if (num_variables[0])
      {
         grid_cmplxty = total_variables / num_variables[0];
      }
      if (num_coeffs[0])
      {
         operat_cmplxty = total_coeffs / num_coeffs[0];
         cycle_cmplxty = cycle_op_count / num_coeffs[0];
      }

      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            hypre_printf("\n\n==============================================");
            hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            hypre_printf("      within the allowed %d V-cycles\n", max_iter);
            hypre_printf("==============================================");
         }
         hypre_printf("\n\n Average Convergence Factor = %f", conv_factor);
         hypre_printf("\n\n     Complexity:    grid = %f\n", grid_cmplxty);
         hypre_printf("                operator = %f\n", operat_cmplxty);
         hypre_printf("                   cycle = %f\n\n\n\n", cycle_cmplxty);
      }

      hypre_TFree(num_coeffs, HYPRE_MEMORY_HOST);
      hypre_TFree(num_variables, HYPRE_MEMORY_HOST);
   }
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}



#ifdef HYPRE_EXVIS
#include "vis.c"
#endif

#define my_min(a,b)  (((a)<(b)) ? (a) : (b))

int main(int argc, char *argv[])
{
   int i;
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   int solver_id;
   int vis, print_system;

   double h, h2;

   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   HYPRE_Solver solver, precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Initialize();

   /* Default problem parameters */
   n = 33;
   solver_id = 0;
   vis = 0;
   print_system = 0;

   /* Preliminaries: want at least one processor per row */
   if (n * n < num_procs)
   {
      n = sqrt(num_procs) + 1;
   }
   N = n * n;         /* global number of rows */
   h = 1.0 / (n + 1); /* mesh size*/
   h2 = h * h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N / num_procs;
   extra = N - local_size * num_procs;

   ilower = local_size * myid;
   ilower += my_min(myid, extra);

   iupper = local_size * (myid + 1);
   iupper += my_min(myid + 1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */
   {
      int nnz;
      /* OK to use constant-length arrays for CPUs
      double values[5];
      int cols[5];
      */
      double *values = (double *)malloc(5 * sizeof(double));
      int *cols = (int *)malloc(5 * sizeof(int));
      int *tmp = (int *)malloc(2 * sizeof(int));

      for (i = ilower; i <= iupper; i++)
      {
         nnz = 0;

         /* The left identity block:position i-n */
         if ((i - n) >= 0)
         {
            cols[nnz] = i - n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The left -1: position i-1 */
         if (i % n)
         {
            cols[nnz] = i - 1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the diagonal: position i */
         cols[nnz] = i;
         values[nnz] = 4.0;
         nnz++;

         /* The right -1: position i+1 */
         if ((i + 1) % n)
         {
            cols[nnz] = i + 1;
            values[nnz] = -1.0;
            nnz++;
         }

         /* The right identity block:position i+n */
         if ((i + n) < N)
         {
            cols[nnz] = i + n;
            values[nnz] = -1.0;
            nnz++;
         }

         /* Set the values for row i */
         tmp[0] = nnz;
         tmp[1] = i;
         HYPRE_IJMatrixSetValues(A, 1, &tmp[0], &tmp[1], cols, values);
      }

      free(values);
      free(cols);
      free(tmp);
   }

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);

   /* Note: for the testing of small problems, one may wish to read
      in a matrix in IJ format (for the format, see the output files
      from the -print_system option).
      In this case, one would use the following routine:
      HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                          HYPRE_PARCSR, &A );
      <filename>  = IJ.A.out to read in what has been printed out
      by -print_system (processor numbers are omitted).
      A call to HYPRE_IJMatrixRead is an *alternative* to the
      following sequence of HYPRE_IJMatrix calls:
      Create, SetObjectType, Initialize, SetValues, and Assemble
   */

   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);

   /* Create the rhs and solution */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   /* Set the rhs values to h^2 and the solution to zero */
   {
      double *rhs_values, *x_values;
      int *rows;

      rhs_values = (double *)calloc(local_size, sizeof(double));
      x_values = (double *)calloc(local_size, sizeof(double));
      rows = (int *)calloc(local_size, sizeof(int));

      for (i = 0; i < local_size; i++)
      {
         rhs_values[i] = h2;
         x_values[i] = 0.0;
         rows[i] = ilower + i;
      }

      HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

      free(x_values);
      free(rhs_values);
      free(rows);
   }

   HYPRE_IJVectorAssemble(b);
   /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
       HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &b );
       as an alternative to the
       following sequence of HYPRE_IJVectors calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
   */
   HYPRE_IJVectorGetObject(b, (void **)&par_b);

   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **)&par_x);

   /*  Print out the system  - files names will be IJ.out.A.XXXXX
        and IJ.out.b.XXXXX, where XXXXX = processor id */
   if (print_system)
   {
      HYPRE_IJMatrixPrint(A, "IJ.out.A");
      HYPRE_IJVectorPrint(b, "IJ.out.b");
   }

   /* Choose a solver and solve the system */

   /* AMG */
   int num_iterations;
   double final_res_norm;

   /* Create solver */
   HYPRE_BoomerAMGCreate(&solver);

   /* Set some parameters (See Reference Manual for more parameters) */
   HYPRE_BoomerAMGSetPrintLevel(solver, 3); /* print solve info + parameters */
   HYPRE_BoomerAMGSetOldDefault(solver);    /* Falgout coarsening with modified classical interpolaiton */
   HYPRE_BoomerAMGSetRelaxType(solver, 3);  /* G-S/Jacobi hybrid relaxation */
   HYPRE_BoomerAMGSetRelaxOrder(solver, 1); /* uses C/F relaxation */
   HYPRE_BoomerAMGSetNumSweeps(solver, 1);  /* Sweeeps on each level */
   HYPRE_BoomerAMGSetMaxLevels(solver, 2); /* maximum number of levels */
   HYPRE_BoomerAMGSetTol(solver, 1e-7);     /* conv. tolerance */

   /* Now setup and solve! */
   HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
   BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
   //HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
   // TODO: replace with custom cycle

   /* Run info - needed logging turned on */
   HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
   HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
   if (myid == 0)
   {
      printf("\n");
      printf("Iterations = %d\n", num_iterations);
      printf("Final Relative Residual Norm = %e\n", final_res_norm);
      printf("\n");
   }

   /* Destroy solver */
   HYPRE_BoomerAMGDestroy(solver);

   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI*/
   MPI_Finalize();

   return (0);
}
