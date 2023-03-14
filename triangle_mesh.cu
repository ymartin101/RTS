
/* ****************** Device Code ******************

 * Geometry node programs
 	* Intersection
 		* After OptiX searches through acceleration structure, test the ray for intersection against appropriate triangles
 		* Determine which triangle is the actual intersected triangle
 		* Interpolate surface normals
 	* Bounding box
 		* Computes axis-aligned bounding boxes

 ************************************************ */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "ray_tracer.h"

using namespace optix;

/* Declare variables */

// Variables with attributes
rtDeclareVariable(double3, normal, attribute normal, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

// User input variables (variables passed from the host)
rtBuffer < uint3, 1 > dbuf_triangles;				// Target's triangle/vertex indices
rtBuffer < double3, 1 > dbuf_triVertices;			// Target's vertices
rtBuffer < double3, 1 > dbuf_normals;				// Target's vertex normals
rtDeclareVariable(bool, d_interpolate_smooth, , );	// Enable/disable interpolation (curved surfaces)


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


// Function to subtract double3s
__device__ double3 operator-(double3 a, double3 b)
{
	return to_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Function to multiply double with double3
__device__ double3 operator*(double a, double3 b)
{
	return to_double3(a * b.x, a * b.y, a * b.z);
}

// Function to compute cross product of two double3s
__device__ double3 crossd3(double3 a, double3 b)
{
	return to_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// Function to compute dot product of two double3s
__device__ double dotd3(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;;
}

// Function to compute length product of double3 and return it as a double
__device__ double lengthd3(double3 in)
{
	return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}

// Function to normalise double3 input
__device__ double3 normalised3(double3 in)
{
	double norm = lengthd3(in);
	return to_double3(in.x/norm, in.y/norm, in.z/norm);
}

// Function to return smallest input double
__device__ double fmind(double a, double b)
{
	return a < b ? a : b;
}

// Function to return smallest input double3
__device__ double3 fmind3(double3 a, double3 b)
{
	return to_double3(fmind(a.x, b.x), fmind(a.y, b.y), fmind(a.z, b.z));
}

// Function to return largest input double
__device__ double fmaxd(double a, double b)
{
	return a > b ? a : b;
}

// Function to return largest input double3
__device__ double3 fmaxd3(double3 a, double3 b)
{
	return to_double3(fmaxd(a.x, b.x), fmaxd(a.y, b.y), fmaxd(a.z, b.z));
}

/* Function to perform intersection tests for doubles and double3s; modified version of function in <optixu/optixu_math_namespace.h> */
__device__ bool intersect_triangle_doubles(const Ray& ray, const double3& p0, const double3& p1, const double3& p2, double3& n,
                                            double&  t, double&  beta, double&  gamma)
{
	const double3 e0 = p1 - p0;
	const double3 e1 = p0 - p2;
	n = crossd3(e1, e0);

	// Note: prevHitPoint here is the ray origin RELATIVE to the intersection point as the OVERALL ORIGIN, i.e. (this ray origin =  intersection point - actual ray origin)
	const double3 e2 = (1/dotd3(n, prd.rayDirection)) * (p0 - prd.prevHitPoint);
	const double3 i = crossd3(prd.rayDirection, e2);
	
	beta = dotd3(i, e1);
	gamma = dotd3(i, e0);
	t = dotd3(n, e2);

	return ( (t < ray.tmax) & (t > ray.tmin) & (beta >= 0.0f) & (gamma >= 0.0f) & (beta + gamma <= 1) );
}


/* Intersection Program */

RT_PROGRAM void intersect(int prim_index) // prim_index is computed by OptiX functions and acceleration structure
{
	rtPrintf("Entering Intersect with prim_index: %d!\n\n", prim_index);

	// Triangle's vertex indices
	unsigned int v_idx0 = dbuf_triangles[prim_index].x;
	unsigned int v_idx1 = dbuf_triangles[prim_index].y;
	unsigned int v_idx2 = dbuf_triangles[prim_index].z;

	// Vertex coordinates relative to the origin
	double3 p0 = dbuf_triVertices[v_idx0];
	double3 p1 = dbuf_triVertices[v_idx1];
	double3 p2 = dbuf_triVertices[v_idx2];

	// // Or this? Should be same...
	// double3 p0 = to_double3(dbuf_triVertices[v_idx0].x, dbuf_triVertices[v_idx0].y, dbuf_triVertices[v_idx0].z);
	// double3 p1 = to_double3(dbuf_triVertices[v_idx1].x, dbuf_triVertices[v_idx1].y, dbuf_triVertices[v_idx1].z);
	// double3 p2 = to_double3(dbuf_triVertices[v_idx2].x, dbuf_triVertices[v_idx2].y, dbuf_triVertices[v_idx2].z);

	// printf("double p0: %e, %e, %e\n", p0.x, p0.y, p0.z);
	
	// Compute ray-triangle intersection
	double3 n;
	double t, beta, gamma;
	if (intersect_triangle_doubles( ray, p0, p1 , p2, n, t, beta, gamma )){ // Modified version of function in <optixu/optixu_math_namespace.h>
		if (rtPotentialIntersection( t ) ) {								// This OptiX function, rtPotentialIntersection, must be used here

			// Find the associated vertex normals
			double3 n0 = dbuf_normals[v_idx0];
			double3 n1 = dbuf_normals[v_idx1];
			double3 n2 = dbuf_normals[v_idx2];

			// If interpolation is enabled
			if (d_interpolate_smooth == true) {

				// If number of vertex normals > number of vertices, assume rect shape; vert_normals (8 for rect) was set to face normals (12 for rect)
				if (dbuf_normals.size() > dbuf_triVertices.size()) {
					normal = dbuf_normals[prim_index];
				}
				else {	// Interpolate vertex normals
					normal = to_double3(n1.x*beta + n2.x*gamma + n0.x*(1.0f - beta - gamma),
										n1.y*beta + n2.y*gamma + n0.y*(1.0f - beta - gamma),
										n1.z*beta + n2.z*gamma + n0.z*(1.0f - beta - gamma));
				}

				// Normalise as a double3 to preserve precision
				normal = normalised3(normal);
			}

			else {
				// Use n by default; locally flat face normal
				normal = normalised3(n);
			}

			// OptiX reporting function; must be used here
			rtReportIntersection(0);
		}
	}
}

/* Bounding Box Program */

RT_PROGRAM void bound (int prim_index, float result[6]) // prim_index is computed by OptiX functions and acceleration structure
{
	rtPrintf("Entering Bound with prim_index: %d!\n\n", prim_index);

	// For the triangle index (prim_index), search the buffer for the associated vertex coordinates
	unsigned int v_idx0 = dbuf_triangles[prim_index].x;
	unsigned int v_idx1 = dbuf_triangles[prim_index].y;
	unsigned int v_idx2 = dbuf_triangles[prim_index].z;

	const double3 v0 = dbuf_triVertices[v_idx0];
	const double3 v1 = dbuf_triVertices[v_idx1];
	const double3 v2 = dbuf_triVertices[v_idx2];
	const double area = lengthd3(crossd3(v1 - v0, v2 - v0));

	// OptiX's utility class aabb ("axis aligned bounding box") is used to compute bounding boxes
	optix::Aabb* aabb = (optix::Aabb*)result;

	// Perform computation
	if (area > 0.0f && !isinf(area)) {							// "isinf" checks if value is infinite; unrelated to sine
		double3 d3min = fmind3( fmind3(v0, v1), v2 );
		double3 d3max = fmaxd3( fmaxd3(v0, v1), v2 );

		// Only use float right at the end; maximise precision; use CUDA function to convert double to float and round "outwards" to closest float;
		// Ensures that every ray that would have hit the "double-precision" geometry will still hit the "float-precision" bounding box
		aabb->m_min = make_float3(__double2float_rd(d3min.x), __double2float_rd(d3min.y), __double2float_rd(d3min.z));
		aabb->m_max = make_float3(__double2float_ru(d3max.x), __double2float_ru(d3max.y), __double2float_ru(d3max.z));
	}
	else
		aabb->invalidate();	// OptiX function to invalidate
}
