
/* ****************** Ray Tracing Simulator (RTS) ******************

 * Authors: Kate Williams (2013), Yaaseen Martin (2020)
 * Contributing authors: Zhongliang Chen, Luis Tirado

 * Utilises NVIDIA OptiX

 * The RTS consists of the following files:
 	* ray_tracer.cuh (header file)
 	* ray_tracer.cpp (host code)
 	* ray_tracer.cu (OptiX kernel; context node programs)
 	* triangle_mesh.cu (OptiX kernel; geometry node programs)
 	* normal_shader.cu (OptiX kernel; material node programs)
 	* CMakeLists.txt (user-defined make lists)

 ************************************************************************************************* */

/* ****************** Host Code ******************

 * Create a graph
	* Context node
		* Create a context
 		* Read in user inputs; create input/output buffers
 		* Set up context node programs: ray generation and miss
	* Geometry node
		* Create a geometry node
		* Set up geometry node programs: bounding box and intersection
 	* Material node
		* Create a material node
 		* Set up material node programs: closest hit
	* Geometry instance node
		* Create a geometry node
		* Attach geometry and material nodes
	* Geometry group node
		* Create a geometry group node
 		* Attach geometry instance nodes
 		* Specify an acceleration structure
 * Launch OptiX kernel
	* Validate and compile
	* Launch
 * Launch ray aggregation kernel
 * Save contents of PRD data (optional)
 * Print timing stats to terminal
 * Clean up and free resources

 ************************************************ */

#include <cmath>
#include "rsworld.cuh"
#include "rsradar.cuh"
#include "rstarget.cuh"
#include "rsplatform.cuh"
#include "rspath.cuh"
#include "rsgeometry.cuh"
#include "rsdebug.cuh"
#include "rsparameters.cuh"
#include "rsantenna.cuh"
#include "rsresponse.cuh"
#include "rsnoise.cuh"
#include "aggregation.cuh"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <optix.h>
#include <sutil.h>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <set>
#include <initializer_list>
#include <limits>
#include <stdexcept>

using std::string;
using std::vector;

const char* const PROGRAM_NAME = "soars";	// Required for PTX processing to work correctly

/* Additional functions */

// Function to get mid-point between two vertices in "sphere"; calculate new vertex in sub-division and normalise it, then add it to the end of v
void getMidPoint(int t1, int t2, vector < vector < double > >& v)
{    
    // Calculate mid-point on unit sphere
    vector < double > pm(3, 0); // Size of 3 with all elements set to zero
    pm[0] = (v[t1][0] + v[t2][0])/2;
    pm[1] = (v[t1][1] + v[t2][1])/2;
    pm[2] = (v[t1][2] + v[t2][2])/2;

    // Normalise the mid-point
    double norm = sqrt(pm[0]*pm[0] + pm[1]*pm[1] + pm[2]*pm[2]);
    pm[0] = pm[0]/norm;
    pm[1] = pm[1]/norm;
    pm[2] = pm[2]/norm;
    
    // Add to vertices list
    v.push_back(pm);
}

// Function to compute the area of a triangle from its three vertices
double triangle_area(vector < double > P1, vector < double > P2, vector < double > P3)
{
    // Calculate side lengths
    double L[3];
    L[0] = sqrt((P1[0] - P2[0])*(P1[0] - P2[0]) + (P1[1] - P2[1])*(P1[1] - P2[1]) + (P1[2] - P2[2])*(P1[2] - P2[2]));
    L[1] = sqrt((P2[0] - P3[0])*(P2[0] - P3[0]) + (P2[1] - P3[1])*(P2[1] - P3[1]) + (P2[2] - P3[2])*(P2[2] - P3[2]));
    L[2] = sqrt((P3[0] - P1[0])*(P3[0] - P1[0]) + (P3[1] - P1[1])*(P3[1] - P1[1]) + (P3[2] - P1[2])*(P3[2] - P1[2]));

    // Calculate area (Heron's formula)
    double S = (L[0] + L[1] + L[2])/2; 
    double A = sqrt(S*(S - L[0])*(S - L[1])*(S - L[2]));

    return A;
}

// Function to perform matrix multiplication
vector < vector < double > > matrix_multiply(vector < vector < double > > M1, vector < vector < double > > M2)
{
    // Create output matrix
    vector < vector < double > > M3(M1.size(), vector < double > (M2[0].size(), 0));

    // Multiply M1 and M2
    for (unsigned int i = 0; i < M1.size(); i++)
    {
        for (unsigned int j = 0; j < M2[0].size(); j++)
        {
              M3[i][j] = 0;
              for (unsigned int k = 0; k < M2.size(); k++)
                 M3[i][j] += M1[i][k] * M2[k][j];
        }
    }

    return M3;
}

// Function to transpose a matrix
vector < vector < double > > matrix_transpose(vector < vector < double > > M1)
{
    // Create matrices
    vector < vector < double > > M2(M1[0].size(), vector < double > (M1.size(), 0));

    // Transpose M1 into M2
    for (unsigned int i = 0; i < M1.size(); i++) {
        for (unsigned int j = 0; j < M1[0].size(); j++)
            M2[j][i] = M1[i][j];
    }

    return M2;
}

// Function to modify vertices due to rotational motion; can also be used to rotate vertex normals
// NOTE: All rotations occur ANTI-CLOCKWISE when looking DOWN the same axis
vector < vector < double > > vertex_rotation(vector < vector < double > > vertices, float yaw, float pitch, float roll)
{
    // Form rotation matrix
    vector < vector < double > > Rx = {{1, 0, 0}, {0, std::cos(roll), -std::sin(roll)}, {0, std::sin(roll), std::cos(roll)}};
    vector < vector < double > > Ry = {{std::cos(pitch), 0, std::sin(pitch)}, {0, 1, 0}, {-std::sin(pitch), 0, std::cos(pitch)}};
    vector < vector < double > > Rz = {{std::cos(yaw), -std::sin(yaw), 0}, {std::sin(yaw), std::cos(yaw), 0}, {0, 0, 1}};
    vector < vector < double > > R_total = matrix_multiply(Rz, matrix_multiply(Ry, Rx));

    // Get rotated vertices
    // vertices[0][0] = 0.5; vertices[0][1] = 0; vertices[0][2] = 0;
    vector < vector < double > > verts_rot = matrix_transpose(matrix_multiply(R_total, matrix_transpose(vertices)));
    // printf("[%e, %e, %e];\n", verts_rot[0][0], verts_rot[0][1], verts_rot[0][2]);

    return verts_rot;
}

// // Function to compute vertex normals for a "rect" mesh
// void get_vertex_normals(vector < vector < double > >& vertices, vector < vector < unsigned int > >& tris, vector < vector < double > >& vert_normals)
// {
    // // Initialise zeros matrix to hold vertex normals of each vertex
    // vert_normals.resize(vertices.size());
    // for (unsigned int i = 0; i < vert_normals.size(); i++)
    //     vert_normals[i].resize(3);
    
    // // Loop through each vertex
    // for (unsigned int i = 0; i < vertices.size(); i++) {
        
    //     // Initialise/reset running total of face area of triangles sharing this vertex
    //     double total_area = 0;

    //     // Loop through triangle matrix rows
    //     for (unsigned int j = 0; j < tris.size(); j++) {

    //         // Loop through triangle matrix columns and find vertex indices
    //         for (unsigned int k = 0; k < tris[0].size(); k++) {

    //             // Check if vertex is used by current triangle
    //             if (i == tris[j][k]) {
                    
    //                 // Calculate the area of the current triangle using its three vertex points
    //                 double tri_area = triangle_area(vertices[tris[j][0]], vertices[tris[j][1]], vertices[tris[j][2]]);

    //                 // Add area-weighted row of face_mat to corresponding vert_normals row
    //                 vert_normals[i][0] = vert_normals[i][0] + tri_area*face_mat[j][0];
    //                 vert_normals[i][1] = vert_normals[i][1] + tri_area*face_mat[j][1];
    //                 vert_normals[i][2] = vert_normals[i][2] + tri_area*face_mat[j][2];
                    
    //                 // Update running total of face area for this vertex
    //                 total_area = total_area + tri_area;

    //                 // Exit innermost loop
    //                 break;
    //             }
    //         }
    //     }

    //     // Divide each row of vert_normals by total area of all triangles sharing vertex; weighted average
    //     vert_normals[i][0] = vert_normals[i][0]/total_area;
    //     vert_normals[i][1] = vert_normals[i][1]/total_area;
    //     vert_normals[i][2] = vert_normals[i][2]/total_area;

    //     // Normalise each row (vector) of vertex normal matrix
    //     double norm = sqrt(vert_normals[i][0]*vert_normals[i][0] + vert_normals[i][1]*vert_normals[i][1] + vert_normals[i][2]*vert_normals[i][2]);
    //     vert_normals[i][0] = vert_normals[i][0]/norm;
    //     vert_normals[i][1] = vert_normals[i][1]/norm;
    //     vert_normals[i][2] = vert_normals[i][2]/norm;
    // }
// }

// Function to compute vertices and vertex normals for a "rect" mesh
void rect_mesh(float w, float h, float d, vector < vector < double > >& vertices, vector < vector < unsigned int > >& tris,
               vector < vector < double > >& vert_normals, float yaw, float pitch, float roll)
{

    // Set up vertices matrix for a 3-D cubic/rectangular mesh
    vertices.resize(8);
    for (unsigned int i = 0; i < vertices.size(); i++)
        vertices[i].resize(3);

    vertices[0][0] = w*+0.5f; vertices[0][1] = h*-0.5f; vertices[0][2] = d*-0.5f;
    vertices[1][0] = w*+0.5f; vertices[1][1] = h*+0.5f; vertices[1][2] = d*-0.5f;
    vertices[2][0] = w*+0.5f; vertices[2][1] = h*-0.5f; vertices[2][2] = d*+0.5f;
    vertices[3][0] = w*+0.5f; vertices[3][1] = h*+0.5f; vertices[3][2] = d*+0.5f;
    vertices[4][0] = w*-0.5f; vertices[4][1] = h*-0.5f; vertices[4][2] = d*-0.5f;
    vertices[5][0] = w*-0.5f; vertices[5][1] = h*+0.5f; vertices[5][2] = d*-0.5f;
    vertices[6][0] = w*-0.5f; vertices[6][1] = h*-0.5f; vertices[6][2] = d*+0.5f;
    vertices[7][0] = w*-0.5f; vertices[7][1] = h*+0.5f; vertices[7][2] = d*+0.5f;

    // Set up triangle face matrix
    tris.resize(12);
    for (unsigned int i = 0; i < tris.size(); i++)
        tris[i].resize(3);

    tris[0][0] = 0;  tris[0][1] = 1;  tris[0][2] = 2;
    tris[1][0] = 1;  tris[1][1] = 3;  tris[1][2] = 2;
    tris[2][0] = 2;  tris[2][1] = 3;  tris[2][2] = 7;
    tris[3][0] = 2;  tris[3][1] = 7;  tris[3][2] = 6;
    tris[4][0] = 1;  tris[4][1] = 7;  tris[4][2] = 3;
    tris[5][0] = 1;  tris[5][1] = 5;  tris[5][2] = 7;
    tris[6][0] = 6;  tris[6][1] = 7;  tris[6][2] = 4;
    tris[7][0] = 7;  tris[7][1] = 5;  tris[7][2] = 4;
    tris[8][0] = 0;  tris[8][1] = 4;  tris[8][2] = 1;
    tris[9][0] = 1;  tris[9][1] = 4;  tris[9][2] = 5;
    tris[10][0] = 2; tris[10][1] = 6; tris[10][2] = 4;
    tris[11][0] = 0; tris[11][1] = 2; tris[11][2] = 4;

    // Get rotated vertices
    vertices = vertex_rotation(vertices, yaw, pitch, roll); // Rotation
    
    // // Get vertex_normals
    // get_vertex_normals(vertices, tris, vert_normals);   // Invoke function to get vertex normals for "rect"

    // Initialise zeros matrix to hold face normals of each triangle
    vector < vector < double > > face_mat(tris.size(), vector < double > (3, 0));

    // For loop to get face normal of each triangle and store in face_mat
    for (unsigned int i = 0; i < face_mat.size(); i++) {

        // Get vertices using triangle matrix and create vectors between points
        double v1[3], v2[3];
        v1[0] = vertices[tris[i][1]][0] - vertices[tris[i][0]][0]; // Vector from P1 to P2; x value
        v1[1] = vertices[tris[i][1]][1] - vertices[tris[i][0]][1]; // Vector from P1 to P2; y value
        v1[2] = vertices[tris[i][1]][2] - vertices[tris[i][0]][2]; // Vector from P1 to P2; z value
        v2[0] = vertices[tris[i][2]][0] - vertices[tris[i][0]][0]; // Vector from P1 to P3; x value
        v2[1] = vertices[tris[i][2]][1] - vertices[tris[i][0]][1]; // Vector from P1 to P3; y value
        v2[2] = vertices[tris[i][2]][2] - vertices[tris[i][0]][2]; // Vector from P1 to P3; z value

        // Cross product to get face normal and store in matrix
        face_mat[i][0] = (v1[1]*v2[2] - v1[2]*v2[1]);
        face_mat[i][1] = (v1[2]*v2[0] - v1[0]*v2[2]);
        face_mat[i][2] = (v1[0]*v2[1] - v1[1]*v2[0]);

        // Normalise each row (vector) of face normal matrix; don't need vertex normals for rect, so just use face normals as "vert_normals"
        double norm = sqrt(face_mat[i][0]*face_mat[i][0] + face_mat[i][1]*face_mat[i][1] + face_mat[i][2]*face_mat[i][2]);
        face_mat[i][0] = face_mat[i][0]/norm;
        face_mat[i][1] = face_mat[i][1]/norm;
        face_mat[i][2] = face_mat[i][2]/norm;
    }

    // Set vert_normals to the face normals for "rect"; don't need to account for curvature/interpolation
    vert_normals = face_mat;
}

// Function to compute vertices and vertex normals for a "sphere" mesh
void sphere_mesh(unsigned int n, float radius, vector < vector < double > >& vertices, vector < vector < unsigned int > >& tris,
                 vector < vector < double > >& vert_normals, float yaw, float pitch, float roll, unsigned int& numTriangles)
{

    // Regular unit icosahedron (12 vertices)
    double t = (1 + sqrt(5)) / 2;
    vector < vector < double > > v = {
        {-1, t, 0},
        {1, t, 0},
        {-1, -t, 0},
        {1, -t, 0},
        {0, -1, t},
        {0, 1, t},
        {0, -1, -t},
        {0, 1, -t},
        {t, 0, -1},
        {t, 0, 1},
        {-t, 0, -1},
        {-t, 0, 1}
    };

    // Normalise vertices to unit size
    for (unsigned int i = 0; i < v.size(); i++) {
        double norm = sqrt(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
        v[i][0] = v[i][0]/norm;
        v[i][1] = v[i][1]/norm;
        v[i][2] = v[i][2]/norm;
    }

    // Regular unit icosahedron (20 faces)
    vector < vector < unsigned int > > f = {
        {0, 11, 5},
        {0, 5, 1},
        {0, 1, 7},
        {0, 7, 10},
        {0, 10, 11},
        {1, 5, 9},
        {5, 11, 4},
        {11, 10, 2},
        {10, 7, 6},
        {7, 1, 8},
        {3, 9, 4},
        {3, 4, 2},
        {3, 2, 6},
        {3, 6, 8},
        {3, 8, 9},
        {4, 9, 5},
        {2, 4, 11},
        {6, 2, 10},
        {8, 6, 7},
        {9, 8, 1}
    };

    // Recursively sub-divide triangle faces
    for (unsigned int gen = 0; gen < n; gen++) {

        vector < vector < unsigned int > > f_(f.size()*4, vector < unsigned int > (3, 0));    // Initialise and set all elements of f_ to zero
        for (unsigned int i = 0; i < f.size(); i++) {   // For each triangle

            int tri[3];
            tri[0] = f[i][0];
            tri[1] = f[i][1];
            tri[2] = f[i][2];

            // Calculate mid-points and add new them to the end of v
            int a = v.size();
            getMidPoint(tri[0], tri[1], v);
            int b = v.size();
            getMidPoint(tri[1], tri[2], v);
            int c = v.size();
            getMidPoint(tri[2], tri[0], v);

            // Generate new subdivision triangles
            int nfc[4][3] = {
                {tri[0], a, c},
                {tri[1], b, a},
                {tri[2], c, b},
                {a, b, c}
            };

            // Replace triangle with sub-division
            for (unsigned int j = 0; j < 4; j++) {
                int idx = (4*i) + j;
                f_[idx][0] = nfc[j][0];
                f_[idx][1] = nfc[j][1];
                f_[idx][2] = nfc[j][2];
            }
        }

        // Update face matrix
        f = f_;
    }

    // "Return" number of triangles used in the mesh
    numTriangles = f.size();

    // Remove duplicate vertices
    std::set < vector < double > > v_unique(v.begin(), v.end());
    vector < int > ix;
    transform(v.begin(), v.end(), 
    back_inserter(ix), [&](vector < double > x) {
        return std::distance(v_unique.begin(), find(v_unique.begin(), v_unique.end(), x)); 
    });
    vector < vector < double > > verts(v_unique.begin(), v_unique.end());

    // Set the (unit vector) vertices after applying rotations
    vertices = vertex_rotation(verts, yaw, pitch, roll); // Rotation of vertices

    // Save normals matrix for this target; for a sphere centred at the origin, vertex normals are the same as the (unit vector) vertices
    vert_normals = vertices;

    // Reassign faces to trimmed vertex list and remove any duplicate faces
    for (unsigned int i = 0; i < f.size(); i++) {
        f[i][0] = ix[f[i][0]];
        f[i][1] = ix[f[i][1]];
        f[i][2] = ix[f[i][2]];
    }
    std::set < vector < unsigned int > > f_unique( f.begin(), f.end() );
    tris.assign( f_unique.begin(), f_unique.end() );

    // Resize the sphere vertices based on its radius; this does NOT affect the vertex normals, which should be unit vectors
    for (unsigned int i = 0; i < vertices.size(); i++) {
        vertices[i][0] *= radius;
        vertices[i][1] *= radius;
        vertices[i][2] *= radius;
    }
}

// Function to compute vertices and vertex normals for a "file" mesh
void file_mesh(string v_file, string n_file, vector < vector < double > >& vertices, vector < vector < unsigned int > >& tris,
               vector < vector < double > >& vert_normals, float yaw, float pitch, float roll)
{
    // Open first file and get number of lines
    std::ifstream inFile(v_file); 
    unsigned int h_numberTriangles = std::count(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>(), '\n');    // Number of lines in file

    // Resize matrices for the two files
    vertices.resize(h_numberTriangles*3);
    vert_normals.resize(h_numberTriangles*3);
    for (unsigned int i = 0; i < vertices.size(); i++)
        vertices[i].resize(3);
    for (unsigned int i = 0; i < vert_normals.size(); i++)
        vert_normals[i].resize(3);

    // Set tris matrix
    tris.resize(h_numberTriangles);
    for (unsigned int i = 0; i < tris.size(); i++) {
        tris[i].resize(3);
        tris[i][0] = i*3 + 0;
        tris[i][1] = i*3 + 1;
        tris[i][2] = i*3 + 2;
    }

    // Populate vector for triangle vertex coordinates
    FILE* fp = fopen(v_file.c_str(), "r");
    if (fp == NULL) {
        printf("Ray tracer error: Cannot open vertex coordinates file!\n"); // Error checker
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < h_numberTriangles; i++) {    // Open file and scan contents
        if (fscanf(fp, "%lf %lf %lf, %lf %lf %lf, %lf %lf %lf,\n", \
            &vertices[3*i][0], \
            &vertices[3*i][1], \
            &vertices[3*i][2], \
            &vertices[3*i+1][0], \
            &vertices[3*i+1][1], \
            &vertices[3*i+1][2], \
            &vertices[3*i+2][0], \
            &vertices[3*i+2][1], \
            &vertices[3*i+2][2]) == EOF)
            exit(EXIT_FAILURE);
    }
    fclose(fp); // Close file

    // Get rotated vertices
    vertices = vertex_rotation(vertices, yaw, pitch, roll); // Rotation

    // Populate vector for triangle vertex normals
    fp = fopen(n_file.c_str(), "r");
    if (fp == NULL) {
        printf("Ray tracer error: Cannot open vertex normals file!\n"); // Error checker
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < h_numberTriangles; i++) {    // Open file and scan contents
        if (fscanf(fp, "%lf %lf %lf, %lf %lf %lf, %lf %lf %lf,\n", \
            &vert_normals[3*i][0], \
            &vert_normals[3*i][1], \
            &vert_normals[3*i][2], \
            &vert_normals[3*i+1][0], \
            &vert_normals[3*i+1][1], \
            &vert_normals[3*i+1][2], \
            &vert_normals[3*i+2][0], \
            &vert_normals[3*i+2][1], \
            &vert_normals[3*i+2][2]) == EOF)
            exit(EXIT_FAILURE);
    }
    fclose(fp); // Close file

    // Get rotated vertex normals; function works for a file mesh's vertex normals too
    // Basically rotating unit vector normals w.r.t. origin
    // Almost like translating, rotating, then translating back; but "new point" does not matter; the VECTOR DIFFERENCE does
    vert_normals = vertex_rotation(vert_normals, yaw, pitch, roll); // Rotation
}


/* Main RTS function */

namespace rs {

    // Define function RTS for ray-tracing implementation; initially declared in rsworld.cuh
    void RTS(World *world, unsigned int MaxThreads, unsigned int MaxBlocks)
    {
        // Timer for RTS set-up runtime
        struct timeval timer1, timer2, timer3;
        gettimeofday(&timer1, NULL);
        double StartTime_PP;
        double StartTime_RTS = timer1.tv_sec + (timer1.tv_usec/1000000.0);

    	// Indicate RTS start
    	printf("Setting up RTS...\n");

        /* *************** CONTEXT SETUP *************** */

        /* Declare context object and program objects */
        RTcontext context;                  // Name of context node
        RTprogram rtprog_ray_generation;    // Name of ray generation program
        RTprogram rtprog_miss;              // Name of miss program
        // RTprogram rtprog_exception;         // Name of exception program

        /* Create context */
        RT_CHECK_ERROR( rtContextCreate( &context ) );
        RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 1 ) );
        RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

        // /* Enable printing for debugging */
        // RT_CHECK_ERROR( rtContextSetPrintEnabled( context, 1) );
        // RT_CHECK_ERROR( rtContextSetPrintBufferSize( context, 1024*1024) );
        // int temp1 = 1; int temp2 = 1; int temp3 = 0;    // Choose thread to print from
        // rtContextSetPrintLaunchIndex(context, temp1,temp2, temp3);

        /* Declare host and program variables, including user inputs. Host variable name (left), device variable name (right) */

        // Variables for OptiX programs
        RTvariable rtvar_width;         // Number of rays; width/height/depth of virtual screen
        RTvariable rtvar_maxReflDepth;  // Number of allowed ray reflections
        RTvariable rtvar_maxRefrDepth;  // Number of allowed ray refractions
        RTvariable rtvar_maxRayTotal;   // Maximum possible number of rays in total (refracted and reflected)
        RTvariable rtvar_rxsize;        // Number of receivers
        RTvariable rtvar_interpolate_smooth;    // Boolean for enabling interpolation of vertex normals
        RTvariable rtvar_rayOrigin;     // Coordinates of point source location
        RTvariable rtvar_txSpan;        // Tx boresight spans and launch range
        RTvariable rtvar_txDir;         // Tx boresight direction
        

        // Set up receiver buffers
        RTbuffer    rtbuf_sphCentre;                                    // Name of buffer that will contain sphCentre
        double3* hbuf_sphCentre;  RTvariable rtvar_sphCentre;           // Variables for sphCentre buffer

        RTbuffer    rtbuf_sphRadius;                                    // Name of buffer that will contain sphRadius
        double* hbuf_sphRadius;  RTvariable rtvar_sphRadius;            // Variables for sphRadius buffer

        RTbuffer    rtbuf_minTheta;                                     // Name of buffer that will contain minTheta
        double* hbuf_minTheta;  RTvariable rtvar_minTheta;              // Variables for minTheta buffer

        RTbuffer    rtbuf_maxTheta;                                     // Name of buffer that will contain maxTheta
        double* hbuf_maxTheta;  RTvariable rtvar_maxTheta;              // Variables for maxTheta buffer

        RTbuffer    rtbuf_minPhi;                                       // Name of buffer that will contain minPhi
        double* hbuf_minPhi;  RTvariable rtvar_minPhi;                  // Variables for minPhi buffer

        RTbuffer    rtbuf_maxPhi;                                       // Name of buffer that will contain maxPhi
        double* hbuf_maxPhi;  RTvariable rtvar_maxPhi;                  // Variables for maxPhi buffer


        // Other buffers
        RTbuffer    rtbuf_triangles;                                    // Name of buffer that will contain triangles
        uint3* hbuf_triangles;  RTvariable rtvar_triangles;             // Variables for triangles buffer

        RTbuffer    rtbuf_triVertices;                                  // Name of buffer that will contain triangle vertices
        double3*     hbuf_triVertices;   RTvariable rtvar_triVertices;  // Variables for verts buffer

        RTbuffer    rtbuf_normals;                                      // Name of buffer that will contain vertex normals
        double3*     hbuf_normals;       RTvariable rtvar_normals;      // Variables for normals buffer

        RTbuffer    rtbuf_results;                                      // Name of buffer that will contain OptiX output results
        PerRayData* hbuf_results;       RTvariable rtvar_results;       // Variables for output results buffer

        RTbuffer    rtbuf_targ_intersect;                               // Name of buffer that will contain OptiX output intersections
        int* hbuf_targ_intersect;    RTvariable rtvar_targ_intersect;   // Variables for output intersections buffer

        RTbuffer    rtbuf_rcs_angle;                                    // Name of buffer that will contain RCS tAngle values
        double2* hbuf_rcs_angle;    RTvariable rtvar_rcs_angle;         // Variables for output RCS angles buffer (azi and ele)

        RTbuffer    rtbuf_targ_vel;                                     // Name of buffer to hold target positions
        double3* hbuf_targ_vel;  RTvariable rtvar_targ_vel;             // Variables for target positions array


        // Set up ray count and reflection/refraction depths
        uint3 rts_vars = rsParameters::GetRTSVariables();    
        unsigned int h_numRays = rts_vars.x;            // Number of rays spawned in each dimension (x, y, z) for each target
        unsigned int h_maxReflDepth = rts_vars.y;       // Maximum number of reflections; user input (max. desired reflections per ray)
        unsigned int h_maxRefrDepth = rts_vars.z;       // Maximum number of refractions
        if (h_maxRefrDepth > 0)                         // Maximum of 2; avoids unnecessary memory usage from a large number of "negligible" refractions
            h_maxRefrDepth = 2;                         // 1 refraction is pointless as the refracted ray would remain "trapped" within target

        // Compute the total number of possible rays (including refractions)
        unsigned int rayTotal = 1;  // Default is 1 (no refractions)
        if (h_maxRefrDepth == 2){   // If there are refractions

            // Possible refractions in each "first-hit object"
            rayTotal += (h_maxReflDepth + 1) + 1;   // Add extra +1 due to ray that becomes "trapped" inside object

            // // Iterate through reflections at the "objects" and account for possible follow-up refractions from the initial refracted ray
            // for (unsigned int i = 0; i <= h_maxReflDepth; i++){

            //     // E.g. For 5 reflections, could have an additional 6 + 5 + 4 + 3 + 2 + 1 = 21 refractions
            //     // Due to refractions caused at the "object" interface by reflections of the initial refracted ray INSIDE the "object"
            //     // i.e. Even after final reflection is reached, the ray can hit another object and refract (but not reflect)
            //     // Reflection can also occurs at a possible final object that is intersected when max reflection depth is reached
            //     rayTotal += (h_maxReflDepth + 1 - i);
            // }
        }

        // Account for total number of rays being transmitted
        rayTotal *= h_numRays*h_numRays*h_numRays;

        /* Create output buffer */
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_results", &rtvar_results ) );  // Create device variable dbuf_results
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &rtbuf_results ) );          // Create buffer
        RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_results, RT_FORMAT_USER ) );
        RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_results, sizeof(PerRayData) ) );          // Set size of buffer
        RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_results, rayTotal ) );                         // Need to account for possible refracted rays
        RT_CHECK_ERROR( rtVariableSetObject( rtvar_results, rtbuf_results ) );                  // Associate contents

        /* Initial set-up */

        // Array and parameter set-up
        unsigned int txsize = (world->transmitters).size();                 // Number of transmitters
        unsigned int rxsize = (world->receivers).size();                    // Number of receivers
        unsigned int targsize = (world->targets).size();                    // Number of targets
        Transmitter** trans_arr = (world->transmitters).data();             // Transmitters array
        Receiver** recv_arr = (world->receivers).data();                    // Receivers array
        Target** targ_arr = (world->targets).data();                        // Targets array
        double cspeed = rsParameters::c();                                  // Speed of propagation
        double sim_starttime = rsParameters::start_time();                  // Start time of the simulation
        double sample_time = 1.0 / rsParameters::cw_sample_rate();          // Default CW sample rate is 1 kHz
        bool h_interpolate_smooth = rsParameters::interpolate_smooth();     // Boolean to enable interpolation of vertex normals
        

        /// Create and fill buffer tracking the paths of all rays
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_targ_intersect", &rtvar_targ_intersect ) );  // Create device variable dbuf_targ_intersect
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &rtbuf_targ_intersect ) ); // Create buffer
        RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_targ_intersect, RT_FORMAT_INT ) );                 // Set buffer to hold ints
        RT_CHECK_ERROR( rtBufferSetSize2D( rtbuf_targ_intersect, (h_maxRefrDepth + h_maxReflDepth), rayTotal ) );   // Set width, height of buffer
        RT_CHECK_ERROR( rtVariableSetObject( rtvar_targ_intersect, rtbuf_targ_intersect ) );        // Associate contents

         /// Create and fill buffer tracking the RCS angles of all rays
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_rcs_angle", &rtvar_rcs_angle ) );  // Create device variable dbuf_rcs_angle
        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT_OUTPUT, &rtbuf_rcs_angle ) );      // Create buffer
        RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_rcs_angle, RT_FORMAT_USER ) );
        RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_rcs_angle, sizeof(double2) ) );               // Set size of buffer
        RT_CHECK_ERROR( rtBufferSetSize2D( rtbuf_rcs_angle, (h_maxRefrDepth + h_maxReflDepth), rayTotal ) );   // Set width, height of buffer
        RT_CHECK_ERROR( rtVariableSetObject( rtvar_rcs_angle, rtbuf_rcs_angle ) );                  // Associate contents


        /// Define receiver buffers

            // sphCentre
            RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_sphCentre", &rtvar_sphCentre ) );      // Declare device variable
            RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_sphCentre ) );                 // Create buffer
            RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_sphCentre, RT_FORMAT_USER ) );                         // Set buffer to hold double3s
            RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_sphCentre, sizeof(double3) ) );                   // Use element size of double3
            RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_sphCentre, rxsize ) );                                 // Set size of buffer
            RT_CHECK_ERROR( rtVariableSetObject( rtvar_sphCentre, rtbuf_sphCentre ) );                      // Associate contents

            // sphRadius
            RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_sphRadius", &rtvar_sphRadius ) );      // Declare device variable
            RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_sphRadius ) );                 // Create buffer
            RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_sphRadius, RT_FORMAT_USER ) );                         // Set buffer to hold double3s
            RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_sphRadius, sizeof(double) ) );                    // Use element size of double3
            RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_sphRadius, rxsize ) );                                 // Set size of buffer
            RT_CHECK_ERROR( rtVariableSetObject( rtvar_sphRadius, rtbuf_sphRadius ) );                      // Associate contents

            // minTheta
            RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_minTheta", &rtvar_minTheta ) );        // Declare device variable
            RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_minTheta ) );                  // Create buffer
            RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_minTheta, RT_FORMAT_USER ) );                          // Set buffer to hold double3s
            RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_minTheta, sizeof(double) ) );                     // Use element size of double3
            RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_minTheta, rxsize ) );                                  // Set size of buffer
            RT_CHECK_ERROR( rtVariableSetObject( rtvar_minTheta, rtbuf_minTheta ) );                        // Associate contents

            // maxTheta
            RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_maxTheta", &rtvar_maxTheta ) );        // Declare device variable
            RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_maxTheta ) );                  // Create buffer
            RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_maxTheta, RT_FORMAT_USER ) );                          // Set buffer to hold double3s
            RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_maxTheta, sizeof(double) ) );                     // Use element size of double3
            RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_maxTheta, rxsize ) );                                  // Set size of buffer
            RT_CHECK_ERROR( rtVariableSetObject( rtvar_maxTheta, rtbuf_maxTheta ) );                        // Associate contents

            // minPhi
            RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_minPhi", &rtvar_minPhi ) );            // Declare device variable
            RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_minPhi ) );                    // Create buffer
            RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_minPhi, RT_FORMAT_USER ) );                            // Set buffer to hold double3s
            RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_minPhi, sizeof(double) ) );                       // Use element size of double3
            RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_minPhi, rxsize ) );                                    // Set size of buffer
            RT_CHECK_ERROR( rtVariableSetObject( rtvar_minPhi, rtbuf_minPhi ) );                            // Associate contents

            // maxPhi
            RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_maxPhi", &rtvar_maxPhi ) );            // Declare device variable
            RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_maxPhi ) );                    // Create buffer
            RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_maxPhi, RT_FORMAT_USER ) );                            // Set buffer to hold double3s
            RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_maxPhi, sizeof(double) ) );                       // Use element size of double3
            RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_maxPhi, rxsize ) );                                    // Set size of buffer
            RT_CHECK_ERROR( rtVariableSetObject( rtvar_maxPhi, rtbuf_maxPhi ) );                            // Associate contents


        /* Set PTX filename for context node programs */
        const char *ptx = sutil::getPtxString( PROGRAM_NAME, "ray_tracer.cu" );

        /* Ray generation program setup*/
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "ray_generation", &rtprog_ray_generation ) );
        RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, 0, rtprog_ray_generation ) );

        // Declare device variables for the ray generation program
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_width", &rtvar_width ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_width, h_numRays ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_maxRayTotal", &rtvar_maxRayTotal ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_maxRayTotal, rayTotal ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_rayOrigin", &rtvar_rayOrigin ) );     // Value set later; changes over time
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_txSpan", &rtvar_txSpan ) );           // Value set later; changes over time
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_txDir", &rtvar_txDir ) );             // Value set later; changes over time
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_rxsize", &rtvar_rxsize ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_rxsize, rxsize ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_interpolate_smooth", &rtvar_interpolate_smooth ) );
        RT_CHECK_ERROR( rtVariableSetUserData(rtvar_interpolate_smooth, sizeof(bool), &h_interpolate_smooth) );

        /* Miss program setup */
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "miss", &rtprog_miss ) );
        RT_CHECK_ERROR( rtContextSetMissProgram( context, 0, rtprog_miss ) );

        // /* Exception program setup */
        // RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx, "exception", &rtprog_exception ) );
        // RT_CHECK_ERROR( rtContextSetExceptionProgram( context, 0, rtprog_exception ) );


        /* *************** GEOMETRY NODE SETUP *************** */

        /* Declare geometry node objects */
        RTgeometry rtnode_geometry;     // Name of geometry node
        RTprogram  rtprog_intersect;    // Name of intersection program
        RTprogram  rtprog_bound;        // Name of bounding box program

        /* Set PTX filename for geometry node programs */
        const char *ptx_triangle_mesh = sutil::getPtxString( PROGRAM_NAME, "triangle_mesh.cu" );


        /* *************** MATERIAL NODE SETUP *************** */

        /* Declare material node objects */
        RTmaterial rtnode_material;
        RTprogram  rtprog_closest_hit;

        /* Create material node */
        RT_CHECK_ERROR( rtMaterialCreate( context, &rtnode_material ) );

        /* Set PTX filename for material node programs */
        const char *ptx_normal_shader = sutil::getPtxString( PROGRAM_NAME, "normal_shader.cu" );

        /* Closest hit program setup */
        RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx_normal_shader, "closest_hit", &rtprog_closest_hit) );
        RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( rtnode_material, 0, rtprog_closest_hit ) );
        
        // Declare device variables for the closest hit program
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_maxReflDepth", &rtvar_maxReflDepth ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_maxReflDepth, (h_maxReflDepth + 1) ) );                 // reflDepth + 1 = "stop index"; max. reflections per ray = (d_maxReflDepth - 1)
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_maxRefrDepth", &rtvar_maxRefrDepth ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_maxRefrDepth, h_maxRefrDepth ) );


        /* *************** GEOMETRY INSTANCE NODE SETUP *************** */

        /* Declare geometry group node objects */
        RTgeometrygroup rtnode_geoGroup;        // Name of geometry group
        RTacceleration  rtnode_geoGroupAcc;     // Name of acceleration
        RTvariable      rtvar_targ_index;       // Target index
        RTvariable      rtvar_targReflCoeff;    // Reflection coefficient
        RTvariable      rtvar_targRefrIndex;    // Refractive index
        RTvariable      rtvar_targets_all;

        /* Declare geometry instance node objects */
        RTgeometryinstance rtnode_geoInst;      // Name of geometry instance

        /* Create geometry group node */
        RT_CHECK_ERROR( rtGeometryGroupCreate( context, &rtnode_geoGroup ) );
        RT_CHECK_ERROR( rtGeometryGroupSetChildCount( rtnode_geoGroup, targsize ) );

        // Declare context variables for all targets
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_targ_vel", &rtvar_targ_vel ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_targets_all", &rtvar_targets_all ) );


        /* *************** START OF TRANSMITTER LOOP *************** */

        // Iterate through all transmitters
        for (unsigned int tx_i = 0; tx_i < txsize; tx_i++){

            // Transmitter and signal set-up
            Transmitter* trans = trans_arr[tx_i];                               // Each transmitter in the loop
            unsigned int pulseCount = trans->GetPulseCount();                   // Number of pulses to transmit
            TransmitterPulse* signal = new TransmitterPulse();                  // Create new signal
            trans->GetPulse(signal, 0);                                         // Define pulse signal
            RadarSignal *wave = signal->wave;                                   // Pulse wave
            double carrier = wave->GetCarrier();                                // Carrier frequency
            double Wl = cspeed/carrier;                                         // Wavelength
            double3 h_rayOrigin;                                                // Initialise transmitter coordinates
            double2 h_txDir;                                                    // Initialise transmitter boresight direction
            double3 h_txSpan = trans->GetTxSpan();                              // Boresight spans and launch range
            RT_CHECK_ERROR( rtVariableSetUserData(rtvar_txSpan, sizeof(double3), &h_txSpan) );

            // Time set-up
            vector < double > start_time_arr(pulseCount);                       // start_time vector varying with pulse number
        
            /// Assign receiver buffer values; these values do NOT change over time, so no need to repeat these later in the time-step loops
            RT_CHECK_ERROR( rtBufferMap(rtbuf_sphRadius, (void **)&hbuf_sphRadius) );   // Map sphRadius buffer
            for (unsigned j = 0; j < rxsize; j++) {

                // Set overall noise temperature for receiver (antenna + external noise); recorded later into responses
                recv_arr[j]->SetNoiseTemperature(wave->GetTemp() + recv_arr[j]->GetNoiseTemperature());

                // Get radius and theta and phi spans
                double3 rxsphere = recv_arr[j]->GetRxSphere();                          // x = radius; y = thetaSpan; z = phiSpan
                hbuf_sphRadius[j] = rxsphere.x;
            }

            // Unmap receiver buffers
            RT_CHECK_ERROR( rtBufferUnmap(rtbuf_sphRadius) );                           // Unmap buffer


            /* *************** START OF TIME-STEP (PULSES) LOOP *************** */

            // Iterate through all pulses
            for (unsigned int k = 0; k < pulseCount; k++)
            {
                // Pulse start time
                trans->GetPulse(signal, k);                                                             // Pulse signal
                start_time_arr[k] = signal->time;

                // Set up time variables
                double time_t = start_time_arr[k];          // Time at the start of the sample
                vector < unsigned int > numUniquePaths;     // Number of unique paths for each Pulse

                // Map target intersect (ray path) buffer; buffer mapped to 1-D array with ALL elements
                RT_CHECK_ERROR( rtBufferMap(rtbuf_targ_intersect, (void **)&hbuf_targ_intersect) );
                for (unsigned int i = 0; i < rayTotal*(h_maxRefrDepth + h_maxReflDepth); i++) {
                    hbuf_targ_intersect[i] = -1;
                }
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_targ_intersect) );                              // Unmap buffer

                // Map RCS angles buffer; buffer mapped to 1-D array with ALL elements
                RT_CHECK_ERROR( rtBufferMap(rtbuf_rcs_angle, (void **)&hbuf_rcs_angle) );
                for (unsigned int i = 0; i < rayTotal*(h_maxRefrDepth + h_maxReflDepth); i++) {
                    
                    // Use default angle of -1000000 (rad) so that it can be checked later
                    hbuf_rcs_angle[i].x = -1000000;    // Uses a half angle approximation; stored as t_angle = inAngle + outAngle (azi)
                    hbuf_rcs_angle[i].y = -1000000;    // Uses a half angle approximation; stored as t_angle = inAngle + outAngle (ele)
                }
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_rcs_angle) );                                   // Unmap buffer

                /// Map receiver buffers
                RT_CHECK_ERROR( rtBufferMap(rtbuf_sphCentre, (void **)&hbuf_sphCentre) );           // Map buffer
                RT_CHECK_ERROR( rtBufferMap(rtbuf_minTheta, (void **)&hbuf_minTheta) );             // Map buffer
                RT_CHECK_ERROR( rtBufferMap(rtbuf_maxTheta, (void **)&hbuf_maxTheta) );             // Map buffer
                RT_CHECK_ERROR( rtBufferMap(rtbuf_minPhi, (void **)&hbuf_minPhi) );                 // Map buffer
                RT_CHECK_ERROR( rtBufferMap(rtbuf_maxPhi, (void **)&hbuf_maxPhi) );                 // Map buffer


                /// Assign time-varying transmitter values
                
                // Set up Tx (ray source) coordinates
                Vec3 trpos = trans->GetPosition(0);     // Tx position does not change with time
                h_rayOrigin.x = trpos.x;                // Actual Tx coordinates
                h_rayOrigin.y = trpos.y;
                h_rayOrigin.z = trpos.z;
                RT_CHECK_ERROR( rtVariableSetUserData(rtvar_rayOrigin, sizeof(double3), &h_rayOrigin) );

                // Set up initial Tx boresight and beamwidth; used to direct the spawned rays
                h_txDir.x = (trans->GetRotation(time_t)).azimuth;
                h_txDir.y = (trans->GetRotation(time_t)).elevation;
                RT_CHECK_ERROR( rtVariableSetUserData(rtvar_txDir, sizeof(double2), &h_txDir) );


                /// Assign receiver buffer values; these values change over time
                for (unsigned j = 0; j < rxsize; j++) {

                    // Get spherical coordinates of sphere centre relative to Rx coordinates as the origin
                    double h_Rx_azimuth = recv_arr[j]->GetRotation(time_t).azimuth;     // Rx boresight azimuth (restricted between -Pi and Pi)
                    double h_Rx_elevation = recv_arr[j]->GetRotation(time_t).elevation; // Rx boresight elevation (restricted between -Pi/2 and Pi/2)

                    // Get Cartesian coordinates of Rx sphere centre from its spherical coordinates; rho is the sphere radius
                    double3 rxsphere = recv_arr[j]->GetRxSphere();  // x = radius; y = thetaSpan; z = phiSpan
                    Vec3 repos = recv_arr[j]->GetPosition(0);       // Rx position does not change with time
                    hbuf_sphCentre[j].x = repos.x + (rxsphere.x * cosf(h_Rx_elevation) * cosf(h_Rx_azimuth));    // Compute x-coordinate of sphere centre; add to Rx position
                    hbuf_sphCentre[j].y = repos.y + (rxsphere.x * cosf(h_Rx_elevation) * sinf(h_Rx_azimuth));    // Compute y-coordinate of sphere centre; add to Rx position
                    hbuf_sphCentre[j].z = repos.z + (rxsphere.x * sinf(h_Rx_elevation));                         // Compute z-coordinate of sphere centre; add to Rx position

                    // Get Rx position in spherical coordinates RELATIVE to the sphere centre as the origin; elevation always between -Pi/2 and Pi/2 since "r" uses +sqrt(...)
                    h_Rx_azimuth = atan2f((repos.y - hbuf_sphCentre[j].y), (repos.x - hbuf_sphCentre[j].x));
                    h_Rx_elevation = atan2f((repos.z - hbuf_sphCentre[j].z), sqrt((repos.x - hbuf_sphCentre[j].x)*(repos.x - hbuf_sphCentre[j].x) + \
                                            (repos.y - hbuf_sphCentre[j].y)*(repos.y - hbuf_sphCentre[j].y)));


                    // Get min and max theta and phi
                    hbuf_minTheta[j] = h_Rx_azimuth - rxsphere.y/2;
                    hbuf_maxTheta[j] = h_Rx_azimuth + rxsphere.y/2;
                    hbuf_minPhi[j] = h_Rx_elevation - rxsphere.z/2;
                    hbuf_maxPhi[j] = h_Rx_elevation + rxsphere.z/2;
                }

                // Unmap receiver buffers
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_sphCentre) );                                               // Unmap buffer
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_minTheta) );                                                // Unmap buffer
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_maxTheta) );                                                // Unmap buffer
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_minPhi) );                                                  // Unmap buffer
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_maxPhi) );                                                  // Unmap buffer


                /* *************** START OF TARGETS LOOP *************** */


                // Set up target position vectors
                vector < double3 > targ_positions(targsize);                                        // Target positions array
                vector < double3 > targ_positions_end(targsize);                                    // Target positions array, one sample later

                // Iterate through all targets
                for (unsigned int targ_i = 0; targ_i < targsize; targ_i++) {

                    // Populate target intersect buffer

                    // Get and save target centre positions
                    targ_positions[targ_i].x = targ_arr[targ_i]->GetPosition(time_t).x;
                    targ_positions[targ_i].y = targ_arr[targ_i]->GetPosition(time_t).y;
                    targ_positions[targ_i].z = targ_arr[targ_i]->GetPosition(time_t).z;

                    // Get and save "next" target centre positions (one sample later); used for Doppler calculation
                    targ_positions_end[targ_i].x = targ_arr[targ_i]->GetPosition(time_t + sample_time).x;
                    targ_positions_end[targ_i].y = targ_arr[targ_i]->GetPosition(time_t + sample_time).y;
                    targ_positions_end[targ_i].z = targ_arr[targ_i]->GetPosition(time_t + sample_time).z;

                    // Create target vertex matrices
                    vector < vector < unsigned int > > tris;    // Target's triangles matrix
                    vector < vector < double > > verts;         // Target's vertices matrix
                    vector < vector < double > > vert_normals;  // Target's vertex normals matrix

                    // Get the initial rotation/orientation of the target (i.e. at time t = 0)
                    float yaw = (targ_arr[targ_i]->GetTargetRotation(0)).yaw;
                    float pitch = (targ_arr[targ_i]->GetTargetRotation(0)).pitch;
                    float roll = (targ_arr[targ_i]->GetTargetRotation(0)).roll;
                
                    /* Apply target shape */

                    // Get target's shape parameters; gets tris, vertices and vert_normals for time t = 0 */
                    string targShape = (targ_arr[targ_i])->GetShape();                              // Shape of target mesh; "rect", "sphere", or "file"

                    // For cubic/rectangular mesh object
                    if (targShape == "rect") {                                                      // Set width, height and depth of "rect" object
                        float w, h, d;
                        (targ_arr[targ_i])->GetRect(w, h, d);
                        rect_mesh(w, h, d, verts, tris, vert_normals, yaw, pitch, roll);            // Get vertex normals and apply rotation at time t = 0
                    }

                    // For spherical mesh object
                    else if (targShape == "sphere") {
                        unsigned int subdivs;                                                       // Number of subdivisions for "sphere" object
                        float radius;                                                               // Sphere radius
                        (targ_arr[targ_i])->GetSphere(subdivs, radius);                             // Modify subdivs and radius
                        unsigned int numTriangles;                                                  // Number of triangles in this "sphere" mesh
                        sphere_mesh(subdivs, radius, verts, tris, vert_normals, \
                                    yaw, pitch, roll, numTriangles);                                // Get vertex normals and apply rotation at time t = 0
                    }

                    // User-defined file mesh
                    else if (targShape == "file") {
                        string v_file, n_file;
                        (targ_arr[targ_i])->GetFile(v_file, n_file);                                // Gets mesh's half-span
                        file_mesh(v_file, n_file, verts, tris, vert_normals, yaw, pitch, roll);     // Get vertex normals and apply rotation at time t = 0
                    }


                    /* Now have tris, verts and vert_normals for current target; unchanging UNLESS there are time-varying target rotations; apply them now */

                    // Check if there are time-varying rotations for the current target
                    if (targ_arr[targ_i]->GetRotating() == true) {

                        // Create backups of the target vertex matrices; used to "reset" rotations performed
                        vector < vector < double > > verts_backup = verts;                      // Backup of target's vertices matrix
                        vector < vector < double > > vert_normals_backup = vert_normals;        // Backup of target's vertex normals matrix

                        // Rotation at t = 0 has already been processed, so only process rotations for time > (simulation start time)
                        if ((targ_arr[targ_i]->GetRotating() == true) && (time_t > sim_starttime)) {
                            double yaw = (targ_arr[targ_i]->GetTargetRotation(time_t)).yaw;
                            double pitch = (targ_arr[targ_i]->GetTargetRotation(time_t)).pitch;
                            double roll = (targ_arr[targ_i]->GetTargetRotation(time_t)).roll;
                            verts = vertex_rotation(verts_backup, yaw, pitch, roll);                 // Apply vertex rotations; uses "initial rotation" of vertices
                            vert_normals = vertex_rotation(vert_normals_backup, yaw, pitch, roll);   // Apply vertex normal rotations; uses "initial rotation" of vertex normals
                        }
                    }

                    /* Displace target's vertex coordinates by target's centre coordinates */
                    for (unsigned int i = 0; i < verts.size(); i++) {
                        verts[i][0] += targ_positions[targ_i].x;
                        verts[i][1] += targ_positions[targ_i].y;
                        verts[i][2] += targ_positions[targ_i].z;
                    }


                    /* *************** GEOMETRY NODE SETUP FOR EACH TARGET *************** */

                    /* Create geometry node */
                    RT_CHECK_ERROR( rtGeometryCreate( context, &rtnode_geometry ) );
                    RT_CHECK_ERROR( rtGeometrySetPrimitiveCount( rtnode_geometry, tris.size() ) );   // Uses number of triangles

                    /* Intersection program setup */
                    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx_triangle_mesh, "intersect", &rtprog_intersect) );
                    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( rtnode_geometry, rtprog_intersect ) );

                    /* Bounding box program setup */
                    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx_triangle_mesh, "bound", &rtprog_bound) );
                    RT_CHECK_ERROR( rtGeometrySetBoundingBoxProgram( rtnode_geometry, rtprog_bound ) );


                    /* *************** GEOMETRY INSTANCE NODE SETUP FOR EACH TARGET *************** */
                        
                    /* Create geometry instance node */
                    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &rtnode_geoInst ) );

                    /* Attach geometry and material nodes */
                    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( rtnode_geoInst, rtnode_geometry ) );
                    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( rtnode_geoInst, 1 ) );
                    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( rtnode_geoInst, 0, rtnode_material ) );

                    // Set up current target's index, reflection coefficient, refraction index, and half-span
                    double targReflCoeff = targ_arr[targ_i]->GetReflCoeff();
                    double targRefrIndex = targ_arr[targ_i]->GetRefrIndex();
                    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targIndex", &rtvar_targ_index ) );            // Declare targ_index
                    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targReflCoeff", &rtvar_targReflCoeff ) );     // Declare reflCoeff
                    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targRefrIndex", &rtvar_targRefrIndex ) );     // Declare refrIndex
                    RT_CHECK_ERROR( rtVariableSet1ui( rtvar_targ_index, targ_i ) );                                                     // Set targ_index
                    RT_CHECK_ERROR( rtVariableSetUserData( rtvar_targReflCoeff, sizeof(double), &targReflCoeff ) );                     // Set reflCoeff
                    RT_CHECK_ERROR( rtVariableSetUserData( rtvar_targRefrIndex, sizeof(double), &targRefrIndex ) );                     // Set refrIndex


                    // /* Add a transform node */
                    // RT_CHECK_ERROR( rtTransformCreate( context, &transforms[targ_i] ) );
                    // RT_CHECK_ERROR( rtTransformSetChild( transforms[targ_i], rtnode_geoGroup ) );
                    // affine_matrix[3] = targ_positions[targ_i].x;    // Translate the target (AND its vertices) to its centre coordinates (determined by SGP4) within OptiX
                    // affine_matrix[7] = targ_positions[targ_i].y;
                    // affine_matrix[11] = targ_positions[targ_i].z;
                    // RT_CHECK_ERROR( rtTransformSetMatrix( transforms[targ_i], 0, affine_matrix, 0 ) );  // Apply transform


                    /* *************** TARGET MATRICES SETUP *************** */
                    
                    /* Create and fill triangle buffer; not required for file mesh */
                    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "dbuf_triangles", &rtvar_triangles ) );  // Declare device variable dbuf_triangles
                    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_triangles ) );                 // Create buffer
                    RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_triangles, RT_FORMAT_UNSIGNED_INT3 ) );                // Set buffer to hold double3s
                    RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_triangles, tris.size() ) );                            // Set size of buffer
                    RT_CHECK_ERROR( rtVariableSetObject( rtvar_triangles, rtbuf_triangles ) );                      // Associate contents
                    RT_CHECK_ERROR( rtBufferMap(rtbuf_triangles, (void **)&hbuf_triangles) );                       // Map buffer

                    // Populate buffer with tris vector
                    for (unsigned j = 0; j < tris.size(); j++) {
                        hbuf_triangles[j].x = tris[j][0];
                        hbuf_triangles[j].y = tris[j][1];
                        hbuf_triangles[j].z = tris[j][2];
                    }
                    RT_CHECK_ERROR( rtBufferUnmap(rtbuf_triangles) );                                               // Unmap buffer


                    /* Create and fill vertex buffer */
                    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "dbuf_triVertices", &rtvar_triVertices ) );  // Declare device variable dbuf_triVertices
                    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_triVertices ) );               // Create buffer
                    RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_triVertices, RT_FORMAT_USER ) );                       // Set buffer to hold double3s
                    RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_triVertices, sizeof(double3) ) );                 // Use element size of double3
                    RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_triVertices, verts.size() ) );                         // Set size of buffer
                    RT_CHECK_ERROR( rtVariableSetObject( rtvar_triVertices, rtbuf_triVertices ) );                  // Associate contents
                    RT_CHECK_ERROR( rtBufferMap(rtbuf_triVertices, (void **)&hbuf_triVertices) );                   // Map buffer

                    // Populate buffer with verts vector
                    for (unsigned j = 0; j < verts.size(); j++) {
                        hbuf_triVertices[j].x = verts[j][0];
                        hbuf_triVertices[j].y = verts[j][1];
                        hbuf_triVertices[j].z = verts[j][2];
                    }
                    RT_CHECK_ERROR( rtBufferUnmap(rtbuf_triVertices) );                                             // Unmap buffer


                    /* Create and fill vertex normal buffer */
                    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "dbuf_normals", &rtvar_normals) );   // Declare device variable dbuf_normals
                    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_normals ) );           // Create buffer
                    RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_normals, RT_FORMAT_USER ) );                   // Set buffer to hold double3s
                    RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_normals, sizeof(double3) ) );             // Use element size of double3
                    RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_normals, vert_normals.size() ) );              // Set size of buffer
                    RT_CHECK_ERROR( rtVariableSetObject( rtvar_normals, rtbuf_normals ) );                  // Associate contents
                    RT_CHECK_ERROR( rtBufferMap(rtbuf_normals, (void **)&hbuf_normals) );                   // Map buffer

                    // Populate buffer with vert_normals vector
                    for (unsigned j = 0; j < vert_normals.size(); j++) {
                        hbuf_normals[j].x = vert_normals[j][0];
                        hbuf_normals[j].y = vert_normals[j][1];
                        hbuf_normals[j].z = vert_normals[j][2];
                    }
                    RT_CHECK_ERROR( rtBufferUnmap(rtbuf_normals) );                                         // Unmap buffer

                    // Add target geometry instance to geometry group
                    RT_CHECK_ERROR( rtGeometryGroupSetChild( rtnode_geoGroup, targ_i, rtnode_geoInst ) );
                }

                /* *************** END OF TARGET LOOP *************** */


                /* *************** GEOMETRY GROUP NODE SETUP *************** */                    

                /* Specify an acceleration structure */
                RT_CHECK_ERROR( rtAccelerationCreate( context, &rtnode_geoGroupAcc ) );
                RT_CHECK_ERROR( rtAccelerationSetBuilder( rtnode_geoGroupAcc, "Bvh" ) );
                RT_CHECK_ERROR( rtAccelerationSetTraverser( rtnode_geoGroupAcc, "Bvh" ) );
                RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( rtnode_geoGroup, rtnode_geoGroupAcc) );
                RT_CHECK_ERROR( rtAccelerationMarkDirty( rtnode_geoGroupAcc ) );

                /* Declare device variables for the geometry group*/
                RT_CHECK_ERROR( rtVariableSetObject( rtvar_targets_all, rtnode_geoGroup ) );

                /* Create and fill target velocities buffer - all targets' velocities at ONE time instance */
                RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_targ_vel ) );              // Create buffer
                RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_targ_vel, RT_FORMAT_USER ) );                      // Set buffer to hold double3s
                RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_targ_vel, sizeof(double3) ) );                // Use element size of double3
                RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_targ_vel, targsize ) );                            // Set size of buffer
                RT_CHECK_ERROR( rtVariableSetObject( rtvar_targ_vel, rtbuf_targ_vel ) );                    // Associate contents
                RT_CHECK_ERROR( rtBufferMap(rtbuf_targ_vel, (void **)&hbuf_targ_vel) );                     // Map buffer

                // Populate buffer with the target velocities
                for (unsigned int i = 0; i < targsize; i++)
                    hbuf_targ_vel[i] = (targ_positions_end[i] - targ_positions[i])/sample_time;             // Velocity over one sample
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_targ_vel) );                                            // Unmap buffer


                /* *************** LAUNCH OPTIX KERNEL *************** */
                
                /* Validate and compile OptiX */
                RT_CHECK_ERROR( rtContextValidate( context ) );
                RT_CHECK_ERROR( rtContextCompile( context ) );

                // Timer for RTS set-up runtime
                gettimeofday(&timer1, NULL);
                double RTS_time = timer1.tv_sec + (timer1.tv_usec/1000000.0) - StartTime_RTS;
                printf("RTS set-up took %lf seconds.\n", RTS_time);

                // Timer for RTS kernel runtime
                gettimeofday(&timer2, NULL);
                double StartTime_kernel = timer2.tv_sec + (timer2.tv_usec/1000000.0);
                
                /* Launch OptiX */
                RT_CHECK_ERROR( rtContextLaunch3D( context, 0 /* entry point */, h_numRays, h_numRays, h_numRays ) );

                // Timer for RTS kernel runtime
                gettimeofday(&timer2, NULL);
                double RTS_kernel_time = timer2.tv_sec + (timer2.tv_usec/1000000.0) - StartTime_kernel;
                printf("RTS kernel took %lf seconds.\n", RTS_kernel_time);


                /* *************** POST-PROCESS OPTIX RESULTS *************** */

                // Timer for post-processing runtime
                gettimeofday(&timer3, NULL);
                StartTime_PP = timer3.tv_sec + (timer3.tv_usec/1000000.0);
                                    
                // Map OptiX output buffers
                RT_CHECK_ERROR( rtBufferMap(rtbuf_results, (void **)&hbuf_results) );
                RT_CHECK_ERROR( rtBufferMap(rtbuf_targ_intersect, (void **)&hbuf_targ_intersect) );
                RT_CHECK_ERROR( rtBufferMap(rtbuf_rcs_angle, (void **)&hbuf_rcs_angle) );

                // Calculate the number of received rays
                unsigned int receivedRays = 0;
                vector < PerRayData > h_rx_results;         // Empty vector to hold PRD results of RECEIVED rays only
                vector < int > h_rx_intersects;             // Empty vector to hold intersection paths for RECEIVED rays only

                // Loop through all PRDs
                for (unsigned int i = 0; i < rayTotal; i++) {
                    
                    // If ray has hit the receiver
                    if (hbuf_results[i].received >= 0) {

                        // Increment received ray total
                        receivedRays++;

                        // Set up transvec, recvvec and repos
                        SVec3 transvec, recvvec;
                        Vec3 repos = (world->receivers)[hbuf_results[i].received]->GetPosition(0);  // Rx position does not change with time

                        // Calculate the value of Pr (exlcuding Pt) for each ray
                        // NOTE: Use repos for receiver point, NOT endpoint - otherwise this messes with GetRotation; endpoint for range ONLY
                        if ((hbuf_results[i].reflDepth == 0) && (hbuf_results[i].refrDepth == 0)) { // If direct transmission
                            transvec = SVec3(d3_to_V3(h_rayOrigin) - repos);
                            recvvec = SVec3(repos - d3_to_V3(h_rayOrigin));
                        }
                        else {                                                                      // If indirect transmission
                            transvec = SVec3(d3_to_V3(hbuf_results[i].firstHitPoint - h_rayOrigin));
                            recvvec = SVec3(d3_to_V3(hbuf_results[i].prevHitPoint) - repos);
                        }

                        // Time delay
                        transvec.length = 1;                                // Normalise
                        recvvec.length = 1;                                 // Normalise
                        double delay = (hbuf_results[i].rayLength)/cspeed;  // (Distance travelled by ray) / c

                        // Target RCS
                        for (unsigned int k = 0; k < (h_maxRefrDepth + h_maxReflDepth); k++) {       // Iterate through all depths
                            unsigned int depth_ray_index = k + i*(h_maxRefrDepth + h_maxReflDepth); // Depth/ray index in array
                            int targ_k = hbuf_targ_intersect[depth_ray_index];
                            h_rx_intersects.push_back(targ_k);

                            // If targ_k is default/empty; RCS ignored if no target was intersected here
                            if (targ_k >= 0) {
                                double targRCS = targ_arr[targ_k]->GetRCS(hbuf_rcs_angle[depth_ray_index].x, hbuf_rcs_angle[depth_ray_index].y, Wl);
                                // printf("RCS: %d, %e\n", targ_k, targRCS);
                                hbuf_results[i].power *= targRCS;               // Multiply power by targets' RCSs
                            }
                        }

                        // Gains
                        double Gt = trans->GetGain(transvec, trans->GetRotation(time_t), Wl);
                        double Gr = (world->receivers)[hbuf_results[i].received]->GetGain(recvvec, \
                                        (world->receivers)[hbuf_results[i].received]->GetRotation(delay + time_t), Wl);

                        // // Temp working; find "baseline" Gains
                        // double3 targpt; targpt.x = 0; targpt.y = 0; targpt.z = 100;
                        // transvec = SVec3(d3_to_V3(targpt - h_rayOrigin));
                        // recvvec = SVec3(d3_to_V3(targpt) - repos);
                        // delay = (transvec.length + recvvec.length)/cspeed;
                        // double Gt1 = trans->GetGain(transvec, trans->GetRotation(time_t), Wl);
                        // double Gr1 = (world->receivers)[hbuf_results[i].received]->GetGain(recvvec, (world->receivers)[hbuf_results[i].received]->GetRotation(delay + time_t), Wl);
                        // printf("Gt: %e, Gr: %e\n", Gt, Gr);

                        // Update power
                        hbuf_results[i].power *= (Wl*Wl*Gt*Gr);     // Power is multiplied by Pt in "rsresponse"
                        // printf("%d, %e,\n", i, hbuf_results[i].power*10000);
                        // printf("[%e, %e, %e];\n", hbuf_results[i].rayDirection.x, hbuf_results[i].rayDirection.y, hbuf_results[i].rayDirection.z);

                        // Update Doppler; currently holds total Doppler velocity = 2Vr; i.e., multiscatter equation comes to Fd = 2Vr/Wl, and we now have 2Vr
                        double Vr = hbuf_results[i].doppler/2;                                      // First divide by 2 to get overall Vr
                        hbuf_results[i].doppler = carrier*(((1 + Vr/cspeed)/(1 - Vr/cspeed)) - 1);  // Insert Vr into POMR equation, then Fd = Fr - Fc (using "- 1")

                        // Allocate hbuf_results entry to h_rx_results
                        h_rx_results.push_back(hbuf_results[i]);
                    }
                }

                printf("Rays: %d,\n", receivedRays);
                
                // If any rays were received, process them further
                if (receivedRays > 0) {

                    // Received-ray vectors for aggregation
                    vector < double > h_npath_vec(receivedRays, 0);
                    vector < double > h_power_vec(receivedRays, 0);
                    vector < double > h_doppler_vec(receivedRays, 0);
                    vector < double > h_delay_vec(receivedRays, 0);
                    vector < double > h_phase_vec(receivedRays, 0);
                    vector < int > h_pathMatch_vec(receivedRays, rayTotal + 1); // Use (rayTotal + 1) to minimise VALID path ray index

                    // Copy vectors to arrays
                    PerRayData* h_rx_results_arr = h_rx_results.data();
                    int* h_rx_intersects_arr = h_rx_intersects.data();
                    double* h_npath_arr = h_npath_vec.data();
                    double* h_power_arr = h_power_vec.data();
                    double* h_doppler_arr = h_doppler_vec.data();
                    double* h_delay_arr = h_delay_vec.data();
                    double* h_phase_arr = h_phase_vec.data();
                    int* h_pathMatch = h_pathMatch_vec.data();

                    rs::kernel_wrapper(h_rx_results_arr, h_rx_intersects_arr, receivedRays, \
                        (h_maxRefrDepth + h_maxReflDepth), MaxThreads, MaxBlocks, cspeed, \
                        carrier, h_npath_arr, h_power_arr, h_doppler_arr, h_delay_arr, h_phase_arr, h_pathMatch);

                    /* *************** ADD RESPONSES TO RECEIVER *************** */

                    // Find unique values/paths in h_pathMatch
                    vector < int > unique_path_rays(h_pathMatch, h_pathMatch + receivedRays);
                    sort(unique_path_rays.begin(), unique_path_rays.end());
                    unique_path_rays.erase(unique(unique_path_rays.begin(), unique_path_rays.end()), unique_path_rays.end());

                    // Append this unique path count to vector
                    numUniquePaths.push_back(unique_path_rays.size());

                    // Track Rx index
                    unsigned int rx;
                    
                    // Loop through all RECEIVED rays
                    for (unsigned int unique_ray = 0; unique_ray < unique_path_rays.size(); unique_ray++) {

                        // Assign index i
                        unsigned int i = unique_path_rays[unique_ray];  // Actual ray index
                        rx = h_rx_results_arr[i].received;              // Receiver index

                        // Delay and phase from arrays
                        double delay = h_delay_arr[i]; // From aggregation
                        double phase = h_phase_arr[i]; // From aggregation

                        // Add InterpPoint
                        InterpPoint point(h_rx_results_arr[i].power, \
                                        time_t + delay, delay, \
                                        h_rx_results_arr[i].doppler, phase, \
                                        (world->receivers)[h_rx_results_arr[i].received]->GetNoiseTemperature());   // Power and Doppler from hbuf_results

                        // Create new response for this point
                        Response* response = new Response(wave, trans);
                        response->AddInterpPoint(point);
                        (world->receivers)[rx]->AddResponse(response);  // Add to correct receiver
                    }
                }

                // Done with output buffers; unmap them
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_results) );
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_targ_intersect) );
                RT_CHECK_ERROR( rtBufferUnmap(rtbuf_rcs_angle) );

                // Timer for post-processing runtime
                gettimeofday(&timer3, NULL);
                double RTS_PP_time = timer3.tv_sec + (timer3.tv_usec/1000000.0) - StartTime_PP;
                printf("Post-processing took %lf seconds.\n", RTS_PP_time);
            }

            /* *************** END OF PULSE LOOP *************** */
        }

        /* *************** END OF TRANSMITTERS LOOP *************** */
        
        /* *************** FREE MEMORY *************** */

        RT_CHECK_ERROR( rtProgramDestroy( rtprog_ray_generation ) );
    	RT_CHECK_ERROR( rtProgramDestroy( rtprog_miss ) );
        RT_CHECK_ERROR( rtProgramDestroy( rtprog_closest_hit ) );
        RT_CHECK_ERROR( rtProgramDestroy( rtprog_bound ) );
        RT_CHECK_ERROR( rtProgramDestroy( rtprog_intersect ) );
        // RT_CHECK_ERROR( rtProgramDestroy( rtprog_exception ) );

    	RT_CHECK_ERROR( rtBufferDestroy( rtbuf_results ) );
    	RT_CHECK_ERROR( rtBufferDestroy( rtbuf_triVertices ) );
    	RT_CHECK_ERROR( rtBufferDestroy( rtbuf_normals ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_triangles ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_targ_vel ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_sphCentre ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_sphRadius ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_maxTheta ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_minPhi ) );
        RT_CHECK_ERROR( rtBufferDestroy( rtbuf_maxPhi ) );

    	RT_CHECK_ERROR( rtContextDestroy( context ) );

    	printf("Exiting RTS...\n");
    }
}
