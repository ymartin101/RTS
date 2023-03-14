
/* ****************** Device Code ******************

* Material node program
* Closest hit
	* After testing the ray for intersection against appropriate triangles
	* Compute hit-points, reflCoeffs, etc
	* Update PRD for intersecting ray
	* Re-launch ray in refraction/reflection direction

************************************************ */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "ray_tracer.h"

using namespace optix;

/* Declare variables */

// Variables with attributes
rtDeclareVariable(float, hit_t, rtIntersectionDistance, );
rtDeclareVariable(uint3, launchIndex, rtLaunchIndex, );
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(double3, normal, attribute normal, );

// User input variables (variables passed from the host)
rtBuffer < PerRayData, 1 > dbuf_results;			// rtBuffer < Type, dim >
rtBuffer < int, 2 > dbuf_targ_intersect;			// rtBuffer < Type, dim >
rtBuffer < double2, 2 > dbuf_rcs_angle;				// rtBuffer < Type, dim >
rtBuffer < double3, 1 > dbuf_targ_vel;				// Target positions
rtDeclareVariable(double3, d_rayOrigin, , );
rtDeclareVariable(unsigned int, d_maxReflDepth, , );
rtDeclareVariable(unsigned int, d_maxRefrDepth, , );
rtDeclareVariable(rtObject, d_targets_all, , );
rtDeclareVariable(double, d_targReflCoeff, , );
rtDeclareVariable(double, d_targRefrIndex, , );
rtDeclareVariable(unsigned int, d_targIndex, , );
rtDeclareVariable(unsigned int, d_width, , );


/* Device functions */

// Function to make double3 variable
__device__ double3 to_double3(double inx, double iny, double inz)
{
	double3 out;
	out.x = inx;
	out.y = iny;
	out.z = inz;
	return out;
}

// Function to convert float3 to double3
__device__ double3 float3_to_double3(float3 in)
{
	double3 out;
	out.x = in.x;
	out.y = in.y;
	out.z = in.z;
	return out;
}

// Function to add double3s
__device__ double3 operator+(double3 a, double3 b)
{
	return to_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// Function to subtract double3s
__device__ double3 operator-(double3 a, double3 b)
{
	return to_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Function to divide double3 by double
__device__ double3 operator/(double3 a, double b)
{
	return to_double3(a.x/b, a.y/b, a.z/b);
}

// Function to compute length product of double3 and return it as a double
__device__ double lengthd3(double3 in)
{
	return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}

// Function to get the squared magnitude of a double3
__device__ double magsquared3(double3 a)
{
	return (a.x*a.x + a.y*a.y + a.z*a.z);
}

// Function to normalise double3 input
__device__ double3 normalised3(double3 in)
{
	double norm = lengthd3(in);
	return to_double3(in.x/norm, in.y/norm, in.z/norm);
}

// Function to normalise float3 input
__device__ float3 normalise_float3(double in1, double in2, double in3)
{
	double norm = lengthd3(to_double3(in1, in2, in3));
	return make_float3(in1/norm, in2/norm, in3/norm);
}

// Function to compute dot product of two double3s
__device__ double dotd3(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;;
}

// Function to convert double Cartesian coordinates to azimuth and elevation
__device__ double2 cart_to_sph(double3 in)
{
	double2 sph;
	sph.x = atan2(in.y, in.x);
	sph.y = atan2(in.z, sqrt(in.x*in.x + in.y*in.y));
	return sph;
}


/* Closest Hit Program */
RT_PROGRAM void closest_hit(void)
{
	// Print statement for debugging; does not print unless enabled in host code
	rtPrintf("Entering closest hit!\n\n");

	// If ray has not previously hit Earth or been received (i.e., ray has not been terminated)
	if ((prd.end == false) && ((prd.refrDepth < d_maxRefrDepth) || (prd.reflDepth < (d_maxReflDepth - 1)))) {

		// Find current ray's index
		unsigned int rayIndex = launchIndex.z*d_width*d_width + launchIndex.y*d_width + launchIndex.x;	// Ray launch index, excluding refractions

		// Add target index to ray path
		if (prd.refrDepth != 1) {					// Only do this for refrDepth of 0 or 2
			uint2 temp;								// Need uint2 to index 2D buffer
			temp.y = rayIndex + prd.maxRayIndex;	// Account for refractions too; y corresponds to height/row
			temp.x = prd.reflDepth + prd.refrDepth;	// x corresponds to width/column
			if (temp.x < (d_maxRefrDepth + d_maxReflDepth - 1))	
				dbuf_targ_intersect[temp] = (int)(d_targIndex);
		}

		// If a ray intersected a triangle (from intersection program), compute hit-point coordinate, ray length, and partial ray length; update PRD
		double3 hitPoint;
		hitPoint.x = (prd.prevHitPoint).x + (double)hit_t*(prd.rayDirection).x;
		hitPoint.y = (prd.prevHitPoint).y + (double)hit_t*(prd.rayDirection).y;
		hitPoint.z = (prd.prevHitPoint).z + (double)hit_t*(prd.rayDirection).z;
		prd.rayLength += hit_t;

		// if (rayIndex == 505063)
		// 	printf("[%.10e, %.10e, %.10e, %d];\n", (hitPoint).x, (hitPoint).y, (hitPoint).z, rayIndex);

		// Update PRD power and firstHitPoint
		if ((prd.reflDepth == 0) && (prd.refrDepth == 0)) {
			prd.firstHitPoint = hitPoint;							// Save first hit-point before any reflection or refraction; does not change again
			double3 TxRange = prd.firstHitPoint - d_rayOrigin;		// From Tx to first-hit target
			if (lengthd3(TxRange) >= SCENE_EPS)						// Must be larger than scene epsilon
				prd.power = 1/((magsquared3(TxRange))*4*M_PI);		// Set power with Tx (4*pi) and squared range from transmitter to first hit-point
			else
				prd.end = true;
		}
		else {
			double3 TargRange = hitPoint - prd.prevHitPoint;		// From previously-hit target to currently-hit target; PRD hit-point not yet updated
			if (lengthd3(TargRange) >= SCENE_EPS_R)					// Must be larger than scene epsilon for reflected rays
				prd.power *= 1/((magsquared3(TargRange))*4*M_PI);	// Update power with Target (4*pi) and squared range from previous target to current target
			else
				prd.end = true;
		}	// Note: If ray does not reflect/refract after this, it is nullified (see end of file) - so this else statement would not matter anyway

		// Update previous hit-point with current hit-point
		prd.prevHitPoint = hitPoint;

		// if ((prd.reflDepth == 0) && (prd.refrDepth == 0)) {
		// 	printf("[%e, %e, %e];\n", hitPoint.x, hitPoint.y, hitPoint.z);
		// }

		// Variables for refraction and reflection
		float3 hitPoint_f3 = make_float3(hitPoint.x, hitPoint.y, hitPoint.z);
		float3 new_direction;			// New direction for refracted ray, then reflected ray

		// Prepare for Doppler computation
		double3 V_targ = dbuf_targ_vel[d_targIndex];             		// Target velocity vector
		double3 k1, k0;													// k1 and k0 vectors (next and previous ray directions)

		// Backup the PRD for a refracted ray
		PerRayData prd_refr = prd;

		// Update the "previous" refraction index with the most recent value
		prd_refr.refrIndex.x = prd_refr.refrIndex.y;

		// If transmission coefficient would be zero, there is no need to compute refraction
		// Only refract on first intersection for this ray (no reflections yet)
		if ( (fabs(d_targReflCoeff) != 1.00000f) && (prd_refr.refrDepth < d_maxRefrDepth) && (prd_refr.reflDepth == 0))	// Use "<" since refrDepth is incremented INSIDE "if" statement
		{
			// Change refractive index ratio depending on current and previous medium of propagation
			if (prd_refr.refrIndex.x == 1) {									// If first medium is vacuum
				prd_refr.refrIndex.y = d_targRefrIndex;							// Second medium will be current target's material (refraction)
			}
			else {																// If first medium is current target's material
				prd_refr.refrIndex.y = 1;										// Second medium will be vacuum (refraction)
			}

			// Calculate index ratio; must be in float form
			float refr_index_ratio = (float)(prd_refr.refrIndex.y/prd_refr.refrIndex.x);			// Index ratio is n2 / n1 = current index / previous index

			// Check if ray actually will refract
			if ( refract( new_direction, ray.direction, normalise_float3(normal.x, normal.y, normal.z), refr_index_ratio ) )
			{
				unsigned int currentRayIndex = prd_refr.maxRayIndex + (d_width*d_width*d_width);	// Current ray index; write to this index of output buffer
				prd_refr.maxRayIndex = currentRayIndex;						// Increment ray index by total rays spawned when new refraction is created

				// Add currently-intersected target's index to this ray's "path chain"; ALWAYS copy ray's first refraction path to next refraction path
				// refrDepth will always be 0 (first hit) at this point; "currentRay" refers to NEXT refraction here
				// ONLY consider rays' FIRST target intersection, NOT subsequent reflections BEFORE refraction has occured for that ray
				// E.g. Ray 0 hits target, refracts; Ray 0 also reflects and hits another target but it will NOT refract there
				if ((prd_refr.refrDepth == 0) && (currentRayIndex == (d_width*d_width*d_width))) {
					uint2 temp;

					// First refracted ray, which will become "trapped" within the target
					for (unsigned int i = 0; i < (d_maxReflDepth + d_maxRefrDepth - 1); i++) {	// "Columns" of targ_intersect matrix
						temp.y = rayIndex + currentRayIndex;			// Index of refracted ray
						temp.x = i;										// Set all depth columns to current target index; "trapped" ray
						dbuf_targ_intersect[temp] = (int)(d_targIndex);	// "Trapped refracted ray" will forever hit the same target
					}

					// Subsequent refracted rays, which will eventually "exit" the target; start at j = 1 to skip first (above) refracted ray
					for (unsigned int j = 0; j < d_maxReflDepth; j++) {		// Any refraction will give rise to d_maxReflDepth refracted rays (total)
						for (unsigned int i = 0; i < (j + 2); i++) {		// "Columns" of targ_intersect matrix
							temp.y = rayIndex + (j + 2)*currentRayIndex;	// Index of refracted ray
							temp.x = i;										// Set relevant depth columns to current target index
							dbuf_targ_intersect[temp] = (int)(d_targIndex);
						}
					}
				}

				// Create new refracted ray; use INCIDENT scene epsilon
				Ray refr_ray = make_Ray( hitPoint_f3, new_direction, 0, SCENE_EPS, RT_DEFAULT_MAX );

				// Update current PRD for refraction; if max reflDepth reached, ALL ray power is transferred to refracted ray (i.e. prd_refr.power *= 1)
				if ((prd_refr.reflDepth + 1) < d_maxReflDepth)			// If maximum reflection depth is not yet being reached with this intersection (incremented later)
					prd_refr.power *= (1 - fabs(d_targReflCoeff));	// Update refracted ray's power with "power loss" (transmission/reflection)
				prd_refr.refrDepth++;

				// Calculate Doppler shift for this target using k1, k0 and V_targ; (k1 - k0) can be minimum of -2 or maximum of +2 (unit vectors)
				// (k1 - k0) corresponds to the term 2cos(B/2) in the bistatic Fd equation
				k0 = normalised3(prd_refr.rayDirection); 					// Normalised k0 (current ray direction) using double3 version
				prd_refr.rayDirection = float3_to_double3(new_direction);	// Update double3 version of ray direction in PRD
				k1 = normalised3(prd_refr.rayDirection);					// Normalised k1 (next ray direction) using double3 version

				// Add target Doppler velocity to running total (Battaglia, 2011); uses (k1 - k0) to account for negative Fd away from radar
				prd_refr.doppler += dotd3(V_targ, (k1 - k0));	// For refractions, this may be zero since k1 could be the same as k0

				// Save refracted RCS angles (for FileTargets)
				uint2 temp_rcs;											// Need uint2 to index 2D buffer
				temp_rcs.y = rayIndex + currentRayIndex;				// Account for refractions too; y corresponds to height/row; overall ray index
				temp_rcs.x = prd_refr.reflDepth + (prd_refr.refrDepth - 1);		// x corresponds to width/column; -1 to "replicate" targ_intersect uint2 above
				double2 k0_sph = cart_to_sph(k0);								// From previous point to current point (as per FERS)
				double2 k1_sph = cart_to_sph(to_double3(-k1.x, -k1.y, -k1.z));	// Reverse direction so that -k1 goes from next point to current point (as per FERS)
				dbuf_rcs_angle[temp_rcs].x = k0_sph.x + k1_sph.x;		// tAngle azi
				dbuf_rcs_angle[temp_rcs].y = k0_sph.y + k1_sph.y;		// tAngle ele
				
				// Recursively call rtTrace for refracted ray
				rtTrace(d_targets_all, refr_ray, prd_refr);

				// When refraction rtTrace finishes ray traversal, save results; for reflected rays, this is done at the end of the ray generation program
				// Need to use currentRayIndex, NOT maxRayIndex since that accounts for all (nested) refracted ray indices
				dbuf_results[rayIndex + currentRayIndex].reflDepth = prd_refr.reflDepth;
				dbuf_results[rayIndex + currentRayIndex].refrDepth = prd_refr.refrDepth;
				dbuf_results[rayIndex + currentRayIndex].rayLength = prd_refr.rayLength;
				dbuf_results[rayIndex + currentRayIndex].firstHitPoint = prd_refr.firstHitPoint;
				dbuf_results[rayIndex + currentRayIndex].prevHitPoint = prd_refr.prevHitPoint;
				dbuf_results[rayIndex + currentRayIndex].power = prd_refr.power;
				dbuf_results[rayIndex + currentRayIndex].doppler = prd_refr.doppler;
				dbuf_results[rayIndex + currentRayIndex].received = prd_refr.received;
				// dbuf_results[rayIndex + currentRayIndex].rayDirection = prd_refr.rayDirection;	// REMOVE
			}
		}

		/// AFTER THE REFRACTIONS HAVE RECURSIVELY COMPLETED PROCESSING
		// Increment reflDepth OUTSIDE/BEFORE "if" statement; essentially counts the number of intersections of each ray
		prd.reflDepth++;

		// Reflected ray will ALWAYS propagate through the "previous" medium before intersection; affects the next reflection/refraction
		prd.refrIndex.y = prd_refr.refrIndex.x;
		prd.refrIndex.x = prd_refr.refrIndex.x;

		// If the number of ray bounces is below "stop index", then recursively call rtTrace to continue ray traversal
		if (prd.reflDepth < d_maxReflDepth) {	// Use "<" so that d_maxReflDepth is the "stop index" at which reflections are stopped; max. reflections per ray = (d_maxReflDepth - 1)
			
			// Compute reflected ray
			new_direction = reflect( ray.direction, normalise_float3(normal.x, normal.y, normal.z) );	// Must be float3 for OptiX function
			Ray refl_ray = make_Ray( hitPoint_f3, new_direction, 0, SCENE_EPS_R, RT_DEFAULT_MAX );
			prd.power *= d_targReflCoeff;		// Update reflected ray's power with "power loss" (transmission/reflection)

			// Calculate Doppler shift for this target using k1, k0 and V_targ; (k1 - k0) can be minimum of -2 or maximum of +2 (unit vectors)
			// (k1 - k0) corresponds to the term 2cos(B/2) in the bistatic Fd equation
			k0 = normalised3(prd.rayDirection); 					// Normalised k0 (current ray direction) using double3 version
			prd.rayDirection = float3_to_double3(new_direction);	// Update double3 version of ray direction in PRD
			k1 = normalised3(prd.rayDirection);						// Normalised k1 (next ray direction) using double3 version

			// Add target Doppler velocity to running total (Battaglia, 2011); uses (k1 - k0) to account for negative Fd away from radar
			// // CHANGE: Test case
			// if (prd.refrDepth > 0) {
			// 	double3 V_targ2;
			// 	V_targ2.x = V_targ.x * 2; V_targ2.y = V_targ.y * 2; V_targ2.z = V_targ.z * 2;
			// 	prd.doppler += dotd3(V_targ2, (k1 - k0));
			// }
			// else
				prd.doppler += dotd3(V_targ, (k1 - k0));
			
			// if (rayIndex == 504910)
			// 	printf("%e\n", prd.doppler);

			// Save reflected RCS angles (for FileTargets)
			uint2 temp_rcs;											// Need uint2 to index 2D buffer
			temp_rcs.y = rayIndex + prd.maxRayIndex;				// maxRayIndex tracks the "refraction set" to which the ray belongs, e.g. ray 0 means maxRayIndex = 0, ray 1000 means maxRayIndex = 1000
			temp_rcs.x = (prd.reflDepth - 1) + prd.refrDepth;				// x corresponds to width/column; -1 to "replicate" targ_intersect uint2 above
			double2 k0_sph = cart_to_sph(k0);								// From previous point to current point (as per FERS)
			double2 k1_sph = cart_to_sph(to_double3(-k1.x, -k1.y, -k1.z));	// Reverse direction so that -k1 goes from next point to current point (as per FERS)
			dbuf_rcs_angle[temp_rcs].x = k0_sph.x + k1_sph.x;		// tAngle azi
			dbuf_rcs_angle[temp_rcs].y = k0_sph.y + k1_sph.y;		// tAngle ele

			// if (prd.reflDepth > 0)
			// 	printf("[%e, %e, %e, %d];\n", new_direction.x, new_direction.y, new_direction.z, prd.reflDepth);

			// Recursively call rtTrace for reflected ray
			rtTrace(d_targets_all, refl_ray, prd);			// Recursively call rtTrace for reflected ray
		}

		// If number of ray reflections and refractions have exceeded the maximum numbers allowed
		if ((prd.reflDepth + 1 >= d_maxReflDepth) && (prd.refrDepth >= d_maxRefrDepth)) {
			prd.end = true;	// Stop ray from being received; treat ray as if its energy is absorbed by the object it just hit
		}
	}
}
