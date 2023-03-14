
/* ****************** Host & Device Code ****************** 
 
 * Header file for aggregation kernel and wrapper
 	
 ************************************************ */
 
 #include <stdio.h>
 #include <cuda.h>
 #include <stdlib.h>
 #include "rsworld.cuh"
 #include <cuda_runtime.h>
 #include <cuComplex.h>
 #include <math.h>
 #include <string.h>
 #include <sys/time.h>
 
 namespace rs {
	 void kernel_wrapper(PerRayData* h_rx_results_arr, int* h_rx_intersects_arr, unsigned int receivedRays, \
		 unsigned int depthTotal, unsigned int MaxThreads, unsigned int MaxBlocks, double cspeed, double carrier, \
		 double* h_npath_arr, double* h_power_arr, double* h_doppler_arr, double* h_delay_arr, double* h_phase_arr, \
		 int* h_pathMatch);
 }
 