
/* ****************** Device Code ******************

 * Context node programs
 	* Ray generation
 		* Initialise PRD, output buffer
 		* Compute ray directions, rays
 		* Call rtTrace
 		* Save results to output buffer
 	* Miss
 		* Determines if the ray is valid
 		* Computes end points if applicable on the receiving array
 		* Computes reflection coefficient of ray
 	* Exception
 		* Can be used for various debugging

 ************************************************ */

#include <optix_world.h>
#include "ray_tracer.h"

using namespace optix;

/* Declare variables */

// Variables with attributes, defined for the first time here
rtDeclareVariable(uint3, launchIndex, rtLaunchIndex, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

// User input variables (variables passed from the host)
rtBuffer < PerRayData, 1 > dbuf_results;			// rtBuffer < Type, dim >
rtBuffer < double3, 1 > dbuf_sphCentre;		// RxSphere centres
rtBuffer < double, 1 > dbuf_sphRadius;		// RxSphere radii
rtBuffer < double, 1 > dbuf_minTheta;		// RxSphere minThetas
rtBuffer < double, 1 > dbuf_maxTheta;		// RxSphere maxThetas
rtBuffer < double, 1 > dbuf_minPhi;			// RxSphere minPhis
rtBuffer < double, 1 > dbuf_maxPhi;			// RxSphere maxPhis
rtDeclareVariable(rtObject, d_targets_all, , );
rtDeclareVariable(unsigned int, d_width, , );
rtDeclareVariable(double3, d_rayOrigin, , );
rtDeclareVariable(double3, d_txSpan, , );
rtDeclareVariable(double2, d_txDir, , );
rtDeclareVariable(unsigned int, d_rxsize, , );
rtDeclareVariable(unsigned int, d_maxRayTotal, , );
rtDeclareVariable(double, d_beamwidth_azi, , );
rtDeclareVariable(double, d_beamwidth_ele, , );


/* Device functions */

// Normalise an angle to the range (-M_PI to +M_PI)
__device__ void normalise_angle(double& angle) 
{
    while ( angle < -M_PI ) angle += 2*M_PI;
    while ( angle >  M_PI ) angle -= 2*M_PI;
}

// Check if a testAngle is between two other angles, a and b
__device__ bool angle_in_range(double testAngle, double a, double b)
{
    a -= testAngle;
    b -= testAngle;
    normalise_angle( a );
    normalise_angle( b );
    if ( a * b >= 0 )
    	return false;
    return fabs( a - b ) < M_PI;
}

// Function to make double3 variable
__device__ double3 to_double3(double inx, double iny, double inz)
{
	double3 out;
	out.x = inx;
	out.y = iny;
	out.z = inz;
	return out;
}

// Function to normalise double3 input
__device__ double3 normalised3(double3 in)
{
	double norm = sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
	return to_double3(in.x/norm, in.y/norm, in.z/norm);
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

// Function to multiply double3 with double
__device__ double3 operator*(double3 a, double b)
{
	return to_double3(a.x * b, a.y * b, a.z * b);
}

// Function to compute cross product of two double3s
__device__ double3 crossd3(double3 a, double3 b)
{
	return to_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Function to get the squared magnitude of a double3
__device__ double magsquared3(double3 a)
{
	return (a.x*a.x + a.y*a.y + a.z*a.z);
}

// Function to compute length product of double3 and return it as a double
__device__ double lengthd3(double3 in)
{
	return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}

// Function to normalise float3 input
__device__ float3 normalise_float3(double in1, double in2, double in3)
{
	double norm = lengthd3(to_double3(in1, in2, in3));
	return make_float3(in1/norm, in2/norm, in3/norm);
}

// Function to convert double spherical coordinates to Cartesian coordinates (unit length)
__device__ double3 sph_to_cart(double azi, double ele)
{
	double3 cart;
	cart.x = cos(azi)*cos(ele);
    cart.y = sin(azi)*cos(ele);
    cart.z = sin(ele);
	return cart;
}


/* Ray Generation Program */

RT_PROGRAM void ray_generation()
{
	// Print statement for debugging; does not print unless enabled in host code
	rtPrintf("Entering ray generation program...\n\n");

	// Loop through target list, create grid nodes for each target, then send rays in the direction of the nodes
	unsigned int rayTotal = d_width*d_width*d_width;
	unsigned int rayIndex = launchIndex.z*d_width*d_width + launchIndex.y*d_width + launchIndex.x;		// Overall ray launch index

    // Find beam start-point relative to boresight centre; uses spherical coordinates (d_txDir: x = azimuth, y = elevation)
	// Uses d_txSpan: x = azimuth span, y = elevation span, z = launch range
	double3 beamStart = sph_to_cart(-d_txSpan.x/2, -d_txSpan.y/2);
	double3 beamEnd = sph_to_cart(d_txSpan.x/2, d_txSpan.y/2);

    // If only one ray is spawned, set the start position to the boresight/beam centre (in that dimension)
	double3 rayDir_d3;
    if (d_width == 1) {											// d_width = "d_height" = "d_depth"
    	rayDir_d3 = sph_to_cart(d_txDir.x, d_txDir.y);			// Unit length; azimuth and elevation are boresight direction
		// printf("[%.10e, %.10e, %.10e];\n", rayDir_d3.x, rayDir_d3.y, rayDir_d3.z);
	}
	else {

		// Compute ray direction using rectangular coordinates
		rayDir_d3.x = beamStart.x + (((beamEnd.x*(1 + d_txSpan.z)) - beamStart.x)/(d_width - 1)) * (launchIndex.x);	// d_txSpan.z is launch range
		rayDir_d3.y = beamStart.y + ((beamEnd.y - beamStart.y)/(d_width - 1)) * (launchIndex.y);
		rayDir_d3.z = beamStart.z + ((beamEnd.z - beamStart.z)/(d_width - 1)) * (launchIndex.z);
		rayDir_d3 = normalised3(rayDir_d3);

		// Form azimuth rotation matrix; NOTE: Right-hand rule applies along z-axis (like yaw)
		double Rot[3][3] = {{cos(d_txDir.x), -sin(d_txDir.x), 0}, \
							{sin(d_txDir.x), cos(d_txDir.x), 0}, \
							{0, 0, 1}};

		// Apply boresight azimuth rotation; matrix multiplication
		double3 rotated; rotated.x = 0; rotated.y = 0; rotated.z = 0;	// Set rotated variable
		rotated.x += Rot[0][0]*rayDir_d3.x + Rot[0][1]*rayDir_d3.y + Rot[0][2]*rayDir_d3.z;
		rotated.y += Rot[1][0]*rayDir_d3.x + Rot[1][1]*rayDir_d3.y + Rot[1][2]*rayDir_d3.z;
		rotated.z += Rot[2][0]*rayDir_d3.x + Rot[2][1]*rayDir_d3.y + Rot[2][2]*rayDir_d3.z;
		rayDir_d3 = normalised3(rotated);

		// Apply "y-axis" azimuth rotation; rotates y-axis by the same azimuth, then elevation rotation is applied using this "new y-axis"
		// All other terms of matrix multiplication is ZERO since x = z = 0; only leaves y-terms as y = 1
		rotated.x = 0; rotated.y = 0; rotated.z = 0;	// Reset rotated variable
		rotated.x += Rot[0][1];
		rotated.y += Rot[1][1];
		rotated.z += Rot[2][1];
		double3 orth_vec = normalised3(rotated);	// Rotated y-axis

		// Form elevation rotation matrix; see: https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
		// NOTE: sin terms' signs are changed from usual formula; reverses elevation direction such that it agrees with the way used in RTS
		double Rot1[3][3] = {{cos(d_txDir.y) + orth_vec.x*orth_vec.x*(1 - cos(d_txDir.y)), orth_vec.x*orth_vec.y*(1 - cos(d_txDir.y)) + orth_vec.z*sin(d_txDir.y), orth_vec.x*orth_vec.z*(1 - cos(d_txDir.y)) - orth_vec.y*sin(d_txDir.y)}, \
							{orth_vec.y*orth_vec.x*(1 - cos(d_txDir.y)) - orth_vec.z*sin(d_txDir.y), cos(d_txDir.y) + orth_vec.y*orth_vec.y*(1 - cos(d_txDir.y)), orth_vec.y*orth_vec.z*(1 - cos(d_txDir.y)) + orth_vec.x*sin(d_txDir.y)}, \
							{orth_vec.z*orth_vec.x*(1 - cos(d_txDir.y)) + orth_vec.y*sin(d_txDir.y), orth_vec.z*orth_vec.y*(1 - cos(d_txDir.y)) - orth_vec.x*sin(d_txDir.y), cos(d_txDir.y) + orth_vec.z*orth_vec.z*(1 - cos(d_txDir.y))}};

		// Apply boresight elevation rotation; matrix multiplication
		rotated.x = 0; rotated.y = 0; rotated.z = 0;	// Reset rotated variable
		rotated.x += Rot1[0][0]*rayDir_d3.x + Rot1[0][1]*rayDir_d3.y + Rot1[0][2]*rayDir_d3.z;
		rotated.y += Rot1[1][0]*rayDir_d3.x + Rot1[1][1]*rayDir_d3.y + Rot1[1][2]*rayDir_d3.z;
		rotated.z += Rot1[2][0]*rayDir_d3.x + Rot1[2][1]*rayDir_d3.y + Rot1[2][2]*rayDir_d3.z;
		rayDir_d3 = rotated;
		// printf("[%.10e, %.10e, %.10e];\n", rayDir_d3.x, rayDir_d3.y, rayDir_d3.z);
	}

	// Spawn a ray; need direction and origin as float3s - possible loss in precision here
	float3 rayDir_f3 = normalise_float3(rayDir_d3.x, rayDir_d3.y, rayDir_d3.z);	// Normalise double3 and THEN convert to float3
	Ray ray = make_Ray(make_float3(d_rayOrigin.x, d_rayOrigin.y, d_rayOrigin.z), rayDir_f3, 0, SCENE_EPS, RT_DEFAULT_MAX);
	
	// Initialize PRD instance
	PerRayData prd;
	prd.reflDepth = 0;
	prd.refrDepth = 0;
	prd.maxRayIndex = 0;
	prd.rayLength = 0;
	prd.rayDirection = rayDir_d3;						// Used as double3 ray direction for precision
	prd.firstHitPoint = make_double3(0.f, 0.f, 0.f);
	prd.prevHitPoint = d_rayOrigin;						// Initially used as ray origin
	prd.refrIndex.x = 1; prd.refrIndex.y = 1;
	prd.power = 0;
	prd.doppler = 0;
	prd.received = -1;
	prd.end = false;

 	// Initialise output buffer
 	for (unsigned int i = 0; i < (d_maxRayTotal/rayTotal); i++) {	// In this file, rayTotal = rays transmitted; maxRayTotal includes all refractions
		dbuf_results[rayIndex + i*rayTotal].reflDepth = 0;
		dbuf_results[rayIndex + i*rayTotal].refrDepth = 0;
		dbuf_results[rayIndex + i*rayTotal].maxRayIndex = 0;
		dbuf_results[rayIndex + i*rayTotal].rayLength = 0;
		dbuf_results[rayIndex + i*rayTotal].rayDirection = make_double3(0, 0, 0);
		dbuf_results[rayIndex + i*rayTotal].firstHitPoint = make_double3(0, 0, 0);
		dbuf_results[rayIndex + i*rayTotal].prevHitPoint = make_double3(0, 0, 0);
		dbuf_results[rayIndex + i*rayTotal].refrIndex.x = 1; dbuf_results[rayIndex + i*rayTotal].refrIndex.y = 1;
		dbuf_results[rayIndex + i*rayTotal].power = 0;
		dbuf_results[rayIndex + i*rayTotal].doppler = 0;
		dbuf_results[rayIndex + i*rayTotal].received = -1;
		dbuf_results[rayIndex + i*rayTotal].end = false;
	}

	// Call OptiX's rtTrace ray traversal function
	rtTrace(d_targets_all, ray, prd);

	// When reflection rtTrace finishes ray traversal, save results; not all variables need to be saved
	dbuf_results[rayIndex].reflDepth = prd.reflDepth;
	dbuf_results[rayIndex].refrDepth = prd.refrDepth;
	dbuf_results[rayIndex].rayLength = prd.rayLength;
	dbuf_results[rayIndex].firstHitPoint = prd.firstHitPoint;
	dbuf_results[rayIndex].prevHitPoint = prd.prevHitPoint;
	dbuf_results[rayIndex].power = prd.power;
	dbuf_results[rayIndex].doppler = prd.doppler;
	dbuf_results[rayIndex].received = prd.received;
	// dbuf_results[rayIndex].rayDirection = prd.rayDirection;	// REMOVE
}


/* Miss Program */

RT_PROGRAM void miss()
{	
	// Print statement for debugging; does not print unless enabled in host code
 	rtPrintf("Entering miss!\n\n");

 	// If ray has not previously hit Earth
 	if (prd.end == false) {

		double A, B, C, discriminant;	// Set up variables for ray-sphere intersections
		double t[2] = {0, 0};

		// Iterate through every receiver and check if there is an intersection with ray
		for (unsigned int Rx_i = 0; Rx_i < d_rxsize; Rx_i++) {

			/* Compute ray-sphere (ray-receiver) intersection:
			Ray = o + td -> origin (x, y, z) + t.direction (x, y, z)
			Ray origin used as the prevHitPoint
			Sphere equation -> (x - cx)² + (y - cy)² + (z - cz)² - r² = 0
			Substitute x = x_Ray, y = y_Ray and z = z_Ray and get quadratic equation to solve for t
			(ox + tdx - cx)² + (oy + tdy - cy)² + (oz + tdz - cz)² - r² = 0
			t² (dx² + dy² + dz²) + \
			t (2(dx(ox - cx) + dy(oy - cy) + dz(oz - cz))) + \
			1 (ox² + oy² + oz² + cx² + cy² + cz² - 2(cx.ox + cy.oy + cz.oz) - r²) = 0
			--> At² + Bt + C = 0 */

			A = ((prd.rayDirection).x)*((prd.rayDirection).x) + ((prd.rayDirection).y)*((prd.rayDirection).y) + ((prd.rayDirection).z)*((prd.rayDirection).z);
			B = 2*((((prd.prevHitPoint).x - dbuf_sphCentre[Rx_i].x)*(prd.rayDirection).x) + \
				   (((prd.prevHitPoint).y - dbuf_sphCentre[Rx_i].y)*(prd.rayDirection).y) + \
				   (((prd.prevHitPoint).z - dbuf_sphCentre[Rx_i].z)*(prd.rayDirection).z));
			C = (prd.prevHitPoint).x*(prd.prevHitPoint).x + (prd.prevHitPoint).y*(prd.prevHitPoint).y + (prd.prevHitPoint).z*(prd.prevHitPoint).z + \
				(dbuf_sphCentre[Rx_i].x*dbuf_sphCentre[Rx_i].x) + \
				(dbuf_sphCentre[Rx_i].y*dbuf_sphCentre[Rx_i].y) + \
				(dbuf_sphCentre[Rx_i].z*dbuf_sphCentre[Rx_i].z) - \
				2*((dbuf_sphCentre[Rx_i].x*(prd.prevHitPoint).x) + (dbuf_sphCentre[Rx_i].y*(prd.prevHitPoint).y) + (dbuf_sphCentre[Rx_i].z*(prd.prevHitPoint).z)) - \
				dbuf_sphRadius[Rx_i]*dbuf_sphRadius[Rx_i];
			discriminant = B*B - 4*A*C;		// Discriminant

			// If roots of quadratic equation are real-valued, ray intersects sphere surface somewhere
			if (discriminant > 0.f) {

				// printf("Ray here!\n");

				// Solve for t (roots of quadratic equation)
				discriminant = sqrt(discriminant);
				t[0] = (-B - discriminant)/(2*A);	// First root
				t[1] = (-B + discriminant)/(2*A);	// Second root
				// printf("t[0]: %e, t[1]: %e\n", t[0], t[1]);

				// Roots loop
				unsigned int received_root = 2; 	// Default of 2; will be either 0 or 1 if ray hits receiver for this root
				for (int i = 0; i < 2; i++) {		// Use BOTH roots of quadratic equation in case there are two intersections

					// If t >= 0, root is valid; ignore for t < 0
					// Also, total ray length must be larger than incident scene epsilon; if not, assume error (monostatic "direct transmission")
					if ((t[i] >= 0) && ((prd.rayLength + t[i]) > SCENE_EPS) && ((prd.rayLength + t[i]) > SCENE_EPS_R)) {

						// printf("Here: %e, %e, %e\n", (prd.prevHitPoint).x, (prd.prevHitPoint).y, (prd.prevHitPoint).z);

						// End-point of a ray on the sphere's surface
						double3 endPoint;
						endPoint.x = (prd.prevHitPoint).x + t[i]*(prd.rayDirection).x;
						endPoint.y = (prd.prevHitPoint).y + t[i]*(prd.rayDirection).y;
						endPoint.z = (prd.prevHitPoint).z + t[i]*(prd.rayDirection).z;

						// Compute spherical angles; use as double for comparison with minima and maxima later
						// Get theta, phi relative to sphere centre; elevation always between -Pi/2 and Pi/2 since "r" uses +sqrt(...)
						double theta = atan2f((endPoint.y - dbuf_sphCentre[Rx_i].y), (endPoint.x - dbuf_sphCentre[Rx_i].x));	// Uses azimuth angle measured from x-axis towards ray vector component in xy-plane
						double phi = atan2f(endPoint.z - dbuf_sphCentre[Rx_i].z, sqrt(((endPoint.y - dbuf_sphCentre[Rx_i].y) * \
											(endPoint.y - dbuf_sphCentre[Rx_i].y)) + ((endPoint.x - dbuf_sphCentre[Rx_i].x) * \
											(endPoint.x - dbuf_sphCentre[Rx_i].x))));	// Uses elevation angle measured from xy-plane towards ray vector

						// Check that phi is within -M_PI/2 and +M_PI/2
			            if ((phi < -M_PI/2)){						// If phi is below -90 deg
			                theta += M_PI;							// Azimuth change
							phi = -M_PI - phi;						// Elevation change (e.g. to -180 - (-95) = -85 deg)
						}

			            if ((phi > M_PI/2)){						// If phi is above +90 deg
			                theta += M_PI;							// Azimuth change
							phi = M_PI - phi;						// Elevation change (e.g. from 180 - 95 = 85 deg)
						}

			            // Set up variables for angle comparisons
			            double d_maxTheta1 = dbuf_maxTheta[Rx_i];
			            double d_minTheta1 = dbuf_minTheta[Rx_i];
			            double d_maxTheta2 = d_maxTheta1;			// Copy of maxTheta1
		            	double d_minTheta2 = d_minTheta1;			// Copy of minTheta1
		            	double d_maxPhi1 = dbuf_maxPhi[Rx_i];
			            double d_minPhi1 = dbuf_minPhi[Rx_i];
			            double d_maxPhi2 = d_maxPhi1;				// Copy of maxPhi1
		            	double d_minPhi2 = d_minPhi1;				// Copy of minPhi1

			            // Check min. and max. bin Phis are within range of -M_PI/2 < Phi < M_PI/2; if not, create second azimuth/elevation regions
			            // Do not need to check maxPhi1 < -M_PI/2 or minPhi1 > M_PI/2; minimum maxPhi1 is -M_PI/2; maximum minPhi1 is M_PI/2
			            if ((d_minPhi1 < -M_PI/2)){					// If minPhi is below -M_PI/2  (e.g. minPhi1 = -95 deg, maxPhi1 = 55 deg)
			                d_maxTheta2 += M_PI;					// Second azimuth region max
							d_minTheta2 += M_PI;					// Second azimuth region min
							d_maxPhi2 = -M_PI - d_minPhi1;			// Second elevation region max (e.g. to -180 - (-95) = -85 deg)
							d_minPhi2 = -M_PI/2;					// Second elevation region min (e.g. from -90 deg)
							d_minPhi1 = -M_PI/2;					// First elevation region min (e.g. from -90 deg); max stays the same (e.g. 55 deg)
						}

			            if ((d_maxPhi1 > M_PI/2)){					// If maxPhi is above +M_PI/2 (e.g. maxPhi1 = 95 deg, minPhi1 = 55 deg)
			                d_maxTheta2 += M_PI;					// Second azimuth region max
							d_minTheta2 += M_PI;					// Second azimuth region min
							d_minPhi2 = M_PI - d_maxPhi1;			// Second elevation region min (e.g. from 180 - 95 = 85 deg)
							d_maxPhi2 = M_PI/2;						// Second elevation region max (e.g. to +90 deg)
							d_maxPhi1 = M_PI/2;						// First elevation region max (e.g. to +90 deg); min stays the same (e.g. 55 deg)
						}

						// Check if the ray hits the part of the Rx sphere we are observing (i.e. the "antenna")
						// MUST test theta/phi TOGETHER; ray could be within an theta range but not the ASSOCIATED phi range --> INVALID RECEIVAL
						// The below IF statements check if angles are INSIDE the angle ranges
						// For only one azimuth/elevation range, the OR below is useless; performs the same check twice since angle2s = angle1s
						if (((angle_in_range(theta, d_minTheta1, d_maxTheta1)) && (angle_in_range(phi, d_minPhi1, d_maxPhi1))) || \
							(angle_in_range(theta, d_minTheta2, d_maxTheta2)) && (angle_in_range(phi, d_minPhi2, d_maxPhi2)))
						{
							// Update received_root as this ray will be captured for this root i
							if (received_root == 2)				// For first root, default is 2; for second root, default is still 2 if first root not captured
								received_root = i;				// Becomes 0 (first root) or 1 (second root) if previous received_root is 2
							else if (t[received_root] > t[i])	// If previous t[received_root] is larger than current t[i], use current t[i] as new t[received_root]
								received_root = i;				// Use shortest path length; only used when capture occurs TWICE (both roots)
							
							// AFTER roots loop, the new received_root will be used for the recorded computations (see below)
							// If both roots worked, the smaller t[i] is used (first capture)
							// If the received_root = 0 is captured and received_root = 1 is not, received_root = 0 is used
							// If the received_root = 1 is captured and received_root = 0 is not, received_root = 1 is used
							// If neither root makes it this far, received_root = 2 is used (nothing is recorded)
						}
					}
				}

				/// AFTER roots loop is done, check if receiver was intersected for EITHER loop
				if (received_root < 2)	{	// At least one root resulted in a ray capture at the receiver

					// Ray terminated after capture
					prd.end = true;

					// Set i to received_root
					unsigned int i = received_root;

					// End-point of ray on the sphere's surface
					double3 endPoint;
					endPoint.x = (prd.prevHitPoint).x + t[i]*(prd.rayDirection).x;
					endPoint.y = (prd.prevHitPoint).y + t[i]*(prd.rayDirection).y;
					endPoint.z = (prd.prevHitPoint).z + t[i]*(prd.rayDirection).z;
					// printf("[%e, %e, %e];\n", endPoint.x, endPoint.y, endPoint.z);

					// Calculate (part of) the received power (narrow-band bistatic radar equation) and Doppler
					double3 RxRange;
					if ((prd.reflDepth == 0) && (prd.refrDepth == 0)) {				// If direct transmission
						RxRange = endPoint - d_rayOrigin;							// Range from Tx to Rx
						if (lengthd3(RxRange) >= SCENE_EPS) {						// Must be larger than scene epsilon
							prd.power = 1/(4*M_PI*4*M_PI*(magsquared3(RxRange)));	// Set power from Tx to Rx
							prd.doppler = 0;										// Zero relative motion --> zero Doppler
							prd.rayLength += t[i];									// Add rayLength
							prd.received = Rx_i;									// Ray has hit a receiver (index Rx_i)
						}
					}
					else {															// If not direct transmission
						RxRange = endPoint - prd.prevHitPoint;						// From last-hit target to Rx
						if (lengthd3(RxRange) >= SCENE_EPS_R) {						// Must be larger than scene epsilon (generally use SCENE_EPS_R here)
							prd.power *= 1/((magsquared3(RxRange))*4*M_PI*4*M_PI);	// Update power with RxRange and Rx (4*pi)
							prd.rayLength += t[i];									// Add rayLength
							prd.received = Rx_i;									// Ray has hit a receiver (index Rx_i)
						}
					}
				}
			}
		}
	}

	/// Scenario when ray hits the Earth (modelled as a sphere)

	// Print statement for debugging; does not print unless enabled in host code
	rtPrintf("This ray had at least one intersection and then entered miss!\n\n");

	// If the ray hits the Earth (or hits a target after reaching its maximum reflections and refractions) before hitting the receiver
	if (prd.end == false) {

		/* Compute ray-Earth (ray-receiver) intersection (same as for Rx sphere)
		Ray = o + td -> origin (x, y, z) + t.direction (x, y, z)
		Sphere equation -> x² + y² + z² - r² = 0; substitute x = x_Ray, y = y_Ray and z = z_Ray and get quadratic equation to solve for t
		(ox + tdx)² + (oy + tdy)² + (oz + tdz)² - r² = 0
		t² (dx² + dy² + dz²) + t (2(ox.dx + oy.dy + oz.dz) + (ox² + oy² + oz² - r²) = 0 --> At² + Bt + C = 0
		Ray overall origin = prd.prevHitPoint -> relative to overall coordinate system's origin */

		double d_earthRadius = 6378136;	// Earth radius [m]
		double A = ((prd.rayDirection).x)*((prd.rayDirection).x) + ((prd.rayDirection).y)*((prd.rayDirection).y) + ((prd.rayDirection).z)*((prd.rayDirection).z);
		double B = 2*((prd.prevHitPoint).x*(prd.rayDirection).x + (prd.prevHitPoint).y*(prd.rayDirection).y + (prd.prevHitPoint).z*(prd.rayDirection).z);
		double C = (prd.prevHitPoint).x*(prd.prevHitPoint).x + (prd.prevHitPoint).y*(prd.prevHitPoint).y + (prd.prevHitPoint).z*(prd.prevHitPoint).z - d_earthRadius*d_earthRadius;
		double discriminant = B*B - 4*A*C;	// Discriminant
		double t[2] = {0, 0};

		// If roots of quadratic equation are real-valued, ray intersects sphere surface somewhere
		if (discriminant > 0.f) {

			// Solve for t (roots of quadratic equation)
			discriminant = sqrt(discriminant);
			t[0] = (-B - discriminant)/(2*A);	// First root
			t[1] = (-B + discriminant)/(2*A);	// Second root

			for (int i = 0; i < 2; i++) {	// Use both roots of quadratic equation in case there are two intersections

				if ((t[i] >= 0) && (prd.rayLength > 0)) {	// If t >= 0, root is valid

					// Terminate the ray after this; ray has hit the Earth
					prd.end = true;

					// Update PRD
					prd.rayLength += t[i];				// Add rayLength

					// Print statement for debugging; may print twice if there are TWO Earth intersections
					// printf("Ray hit the Earth; rayLength = %e\n", prd.rayLength);
				}
			}
		}
	}
}

// // Exception Program
// RT_PROGRAM void exception() {
// 		rtPrintf("Entering exception program!\n\n");
// 		prd.rayLength = -1.f;
// 		const unsigned int code = rtGetExceptionCode();
// 		rtPrintf( "Caught exception 0x%X at launch index (%d, %d)\n", code, launchIndex.x, launchIndex.y );
// }
