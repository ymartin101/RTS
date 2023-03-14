
/* ****************** Host & Device Code ****************** 

* Error checker
* Device code
* Computes ray aggregation (power scaling)
* Host code
* Set up code for kernel launch

************************************************ */

#include "aggregation.cuh"

/* *************** ERROR CHECKER *************** */

// Check for CUDA errors
#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
				msg, cudaGetErrorString(__err), \
				__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n\n"); \
			exit(1); \
		} \
	} while (0)


/* *************** DEVICE CODE: RAY AGGREGATION KERNEL 1 *************** */
					
extern "C" __global__ void myKernel1(PerRayData* d_results_arr, int* d_targ_intersect_arr, unsigned int receivedRays, \
	unsigned int depthTotal, unsigned int MaxThreads, unsigned int MaxBlocks, double* d_npath_arr, \
	double* d_power_arr, double* d_doppler_arr, double* d_delay_arr, double* d_phase_arr, double cspeed, \
	double carrier, int* d_pathMatch)
{
	// Only use threads up until the end of receivedRays
	int tid = int(threadIdx.x + (blockIdx.x * blockDim.x));	// Thread index
	int stride = blockDim.x * gridDim.x;  						// Total number of threads spawned
	for (int i = tid; i < receivedRays; i += stride){			// Will only iterate again if receivedRays is more than (MaxBlocks*MaxThreads)

		// Iterate through all RECEIVED rays and compare to ray i
		for (unsigned int r = 0; r < receivedRays; r++) {

			// FIRST check for same Rx (to cut down on computations)
			if (d_results_arr[i].received == d_results_arr[r].received) {
					bool row_equal = true;
					for(unsigned int k = 0; k < depthTotal; k++) { // Find ray path for ray r
						if (d_targ_intersect_arr[k + i*depthTotal] != d_targ_intersect_arr[k + r*depthTotal]) {
							row_equal = false;	// Rows are not equal, so immediately abandon this loop
							break;
						}
					}

				// Check/compare paths for ray i and ray r (and check for direct transmission)
				if ((row_equal == true) || ((d_results_arr[i].reflDepth == 0) && (d_results_arr[i].refrDepth == 0))) {
						
					// Update ray quantity totals
					double delay = (d_results_arr[r].rayLength)/cspeed;
					double phase = -fmod(delay*2*M_PI*carrier, 2*M_PI);
					d_npath_arr[i] += 1;											// Increment for matching paths
					d_power_arr[i] += sqrt(d_results_arr[r].power);					// Add "voltages" for common-path rays
					d_delay_arr[i] += delay;										// Add delays for common-path rays
					d_phase_arr[i] += phase;										// Add phases for common-path rays
					d_doppler_arr[i] += d_results_arr[r].doppler;					// Add Dopplers for common-path rays
					
					// Record current ray index as this path's "path index"
					if (r < d_pathMatch[i])		// Always record the "earliest" ray index that follows this path
						d_pathMatch[i] = r;		// Note: iterating through rays "r" will include ray "i" as well
				}
			}
		}
	}
}
 

/* *************** DEVICE CODE: RAY AGGREGATION KERNEL 2 *************** */
					
extern "C" __global__ void myKernel2(PerRayData* d_results_arr, unsigned int receivedRays, double* d_npath_arr, \
	double* d_power_arr, double* d_doppler_arr, double* d_delay_arr, double* d_phase_arr)
{
	// Only use threads up until the end of receivedRays
	int tid = int(threadIdx.x + (blockIdx.x * blockDim.x));	// Thread index
	int stride = blockDim.x * gridDim.x;  						// Total number of threads spawned
	for (int i = tid; i < receivedRays; i += stride){			// Will only iterate again if receivedRays is more than (MaxBlocks*MaxThreads)

		// Divide totals of quantities by Npath (for same-path rays); done in Kernel 2 to avoid hbuf etc being overwritten during Kernel 1 processing
		if (d_npath_arr[i] > 0) {
			d_results_arr[i].power = pow(d_power_arr[i]/d_npath_arr[i], 2);	// Divide total voltage by Npath (averaging), then square
			d_delay_arr[i] /= d_npath_arr[i];									// Divide total delay by Npath
			d_phase_arr[i] /= d_npath_arr[i];									// Divide total phase by Npath
			d_results_arr[i].doppler = d_doppler_arr[i]/d_npath_arr[i];		// Divide total Doppler by Npath
			
			// printf("Power: %e, Npath: %lf\n", d_results_arr[i].power, d_npath_arr[i]);
		}
	}
}


/* *************** HOST CODE: RAY AGGREGATION SETUP *************** */

namespace rs {
	void kernel_wrapper(PerRayData* h_rx_results_arr, int* h_rx_intersects_arr, unsigned int receivedRays, \
		unsigned int depthTotal, unsigned int MaxThreads, unsigned int MaxBlocks, double cspeed, double carrier, \
		double* h_npath_arr, double* h_power_arr, double* h_doppler_arr, double* h_delay_arr, double* h_phase_arr, \
		int* h_pathMatch)
	{ 	   
		// Variables to send to device
		PerRayData* d_results_arr;
		int* d_targ_intersect_arr;
		double* d_npath_arr; double* d_power_arr; double* d_doppler_arr; double* d_delay_arr; double* d_phase_arr;
		int* d_pathMatch;	// Tracks path matches

		// Allocate memory on the device
		cudaMalloc((void **)&d_results_arr, sizeof(PerRayData)*receivedRays);
		cudaMalloc((void **)&d_targ_intersect_arr, sizeof(int)*receivedRays*depthTotal);
		cudaMalloc((void **)&d_power_arr, sizeof(double)*receivedRays);
		cudaMalloc((void **)&d_doppler_arr, sizeof(double)*receivedRays);
		cudaMalloc((void **)&d_delay_arr, sizeof(double)*receivedRays);
		cudaMalloc((void **)&d_phase_arr, sizeof(double)*receivedRays);
		cudaMalloc((void **)&d_npath_arr, sizeof(double)*receivedRays);
		cudaMalloc((void **)&d_pathMatch, sizeof(int)*receivedRays);
		cudaCheckErrors("Malloc fail");

		// Copy inputs to device
		cudaMemcpy(d_results_arr, h_rx_results_arr, sizeof(PerRayData)*receivedRays, cudaMemcpyHostToDevice);
		cudaMemcpy(d_targ_intersect_arr, h_rx_intersects_arr, sizeof(int)*receivedRays*depthTotal, cudaMemcpyHostToDevice);
		cudaMemcpy(d_npath_arr, h_npath_arr, sizeof(double)*receivedRays, cudaMemcpyHostToDevice);
		cudaMemcpy(d_power_arr, h_power_arr, sizeof(double)*receivedRays, cudaMemcpyHostToDevice);
		cudaMemcpy(d_doppler_arr, h_doppler_arr, sizeof(double)*receivedRays, cudaMemcpyHostToDevice);
		cudaMemcpy(d_delay_arr, h_delay_arr, sizeof(double)*receivedRays, cudaMemcpyHostToDevice);
		cudaMemcpy(d_phase_arr, h_phase_arr, sizeof(double)*receivedRays, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pathMatch, h_pathMatch, sizeof(int)*receivedRays, cudaMemcpyHostToDevice);
		cudaCheckErrors("Memory (device) fail");

		// Timer for ray aggregation kernels runtime
		struct timeval timer4;
		gettimeofday(&timer4, NULL);
		double StartTime_RA = timer4.tv_sec + (timer4.tv_usec/1000000.0);

		// Ray aggregation kernel 1 launch
		if (receivedRays <= MaxThreads)				// If there are fewer rays than the number of threads in one block (or an equal number of targets)
			myKernel1<<<1, receivedRays>>>(d_results_arr, d_targ_intersect_arr, receivedRays, depthTotal, MaxThreads, \
				MaxBlocks, d_npath_arr, d_power_arr, d_doppler_arr, d_delay_arr, d_phase_arr, cspeed, carrier, d_pathMatch);
		else if (receivedRays > (MaxThreads*MaxBlocks))	// If there are more rays than the maximum number of parallel threads (across all blocks)
			myKernel1<<<MaxBlocks, MaxThreads>>>(d_results_arr, d_targ_intersect_arr, receivedRays, depthTotal, MaxThreads, \
				MaxBlocks, d_npath_arr, d_power_arr, d_doppler_arr, d_delay_arr, d_phase_arr, cspeed, carrier, d_pathMatch);
		else										// If number of rays requires more than 1 block, but not all of them
			myKernel1<<<((receivedRays + (MaxThreads - 1))/MaxThreads), MaxThreads>>>(d_results_arr, d_targ_intersect_arr, \
				receivedRays, depthTotal, MaxThreads, MaxBlocks, d_npath_arr, \
				d_power_arr, d_doppler_arr, d_delay_arr, d_phase_arr, cspeed, carrier, d_pathMatch);
		cudaCheckErrors("Kernel 1 fail");

		// Ray aggregation kernel 2 launch
		if (receivedRays <= MaxThreads)				// If there are fewer rays than the number of threads in one block (or an equal number of targets)
			myKernel2<<<1, receivedRays>>>(d_results_arr, receivedRays, d_npath_arr, d_power_arr, d_doppler_arr, d_delay_arr, d_phase_arr);
		else if (receivedRays > (MaxThreads*MaxBlocks))	// If there are more rays than the maximum number of parallel threads (across all blocks)
			myKernel2<<<MaxBlocks, MaxThreads>>>(d_results_arr, receivedRays, d_npath_arr, d_power_arr, d_doppler_arr, d_delay_arr, d_phase_arr);
		else										// If number of rays requires more than 1 block, but not all of them
			myKernel2<<<((receivedRays + (MaxThreads - 1))/MaxThreads), MaxThreads>>>(d_results_arr, receivedRays, d_npath_arr, d_power_arr, d_doppler_arr, d_delay_arr, d_phase_arr);
		cudaCheckErrors("Kernel 2 fail");

		// Timer for ray aggregation runtime
        gettimeofday(&timer4, NULL);
        double RTS_RA_time = timer4.tv_sec + (timer4.tv_usec/1000000.0) - StartTime_RA;
        printf("Ray aggregation took %lf seconds.\n", RTS_RA_time);

		// Copy from device 
		cudaMemcpy(h_rx_results_arr, d_results_arr, sizeof(PerRayData)*receivedRays, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_delay_arr, d_delay_arr, sizeof(double)*receivedRays, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_phase_arr, d_phase_arr, sizeof(double)*receivedRays, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_pathMatch, d_pathMatch, sizeof(int)*receivedRays, cudaMemcpyDeviceToHost);
		cudaCheckErrors("Memory (host) fail");

		// Free memory
		cudaFree(d_results_arr);
		cudaFree(d_targ_intersect_arr);
		cudaFree(d_delay_arr);
		cudaFree(d_phase_arr);
		cudaFree(d_npath_arr);
		cudaCheckErrors("Delete fail");

		return;
	}
}
