/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Member functions for AMGData structure
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * amg_NewData
 *--------------------------------------------------------------------------*/

AMGData   *amg_NewData(levmax, ncg, ecg, nwt, ewt, nstr,
		       ncyc, mu, ntrlx, iprlx,
		       ioutdat, cycle_op_count,
		       log_file_name)
int     levmax;
int     ncg;
double  ecg;
int     nwt;
double  ewt;
int     nstr;
int     ncyc;
int    *mu;
int    *ntrlx;
int    *iprlx;
int     ioutdat;
int     cycle_op_count;
char   *log_file_name;
{
   AMGData  *amg_data;

   amg_data = ctalloc(AMGData, 1);

   AMGDataLevMax(amg_data)  = levmax;
   AMGDataNCG(amg_data)     = ncg;
   AMGDataECG(amg_data)     = ecg;
   AMGDataNWT(amg_data)     = nwt;
   AMGDataEWT(amg_data)     = ewt;
   AMGDataNSTR(amg_data)    = nstr;
   				    
   AMGDataNCyc(amg_data)    = ncyc;
   AMGDataMU(amg_data)      = mu;
   AMGDataNTRLX(amg_data)   = ntrlx;
   AMGDataIPRLX(amg_data)   = iprlx;
   				    
   AMGDataIOutDat(amg_data) = ioutdat;
   AMGDataCycleOpCount(amg_data) = cycle_op_count;

   sprintf(AMGDataLogFileName(amg_data), "%s", log_file_name); 

   return amg_data;
}

/*--------------------------------------------------------------------------
 * amg_FreeData
 *--------------------------------------------------------------------------*/

void      amg_FreeData(amg_data)
AMGData  *amg_data;
{
   if (amg_data)
   {
      /* solve params */
      tfree(AMGDataMU(amg_data));
      tfree(AMGDataNTRLX(amg_data));
      tfree(AMGDataIPRLX(amg_data));

      /* problem data */
      /* data generated by the setup phase */

      tfree(amg_data);
   }
}

