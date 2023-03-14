
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
vector < vector < double > > vertex_rotation(vector < vector < double > > vertices, float yaw, float pitch, float roll)
{
    // Form rotation matrix
    vector < vector < double > > Rx = {{1, 0, 0}, {0, std::cos(roll), std::sin(roll)}, {0, -std::sin(roll), std::cos(roll)}};
    vector < vector < double > > Ry = {{std::cos(pitch), 0, -std::sin(pitch)}, {0, 1, 0}, {std::sin(pitch), 0, std::cos(pitch)}};
    vector < vector < double > > Rz = {{std::cos(yaw), std::sin(yaw), 0}, {-std::sin(yaw), std::cos(yaw), 0}, {0, 0, 1}};
    vector < vector < double > > R_total = matrix_multiply(Rz, matrix_multiply(Ry, Rx));

    // Get rotated vertices
    vector < vector < double > > verts_rot = matrix_transpose(matrix_multiply(R_total, matrix_transpose(vertices)));

    return verts_rot;
}

// Function to compute vertex normals for a "rect" mesh
void get_vertex_normals(vector < vector < double > >& vertices, vector < vector < unsigned int > >& tris, vector < vector < double > >& vert_normals)
{

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

        // Normalise each row (vector) of face normal matrix
        double norm = sqrt(face_mat[i][0]*face_mat[i][0] + face_mat[i][1]*face_mat[i][1] + face_mat[i][2]*face_mat[i][2]);
        face_mat[i][0] = face_mat[i][0]/norm;
        face_mat[i][1] = face_mat[i][1]/norm;
        face_mat[i][2] = face_mat[i][2]/norm;
    }

    // Initialise zeros matrix to hold vertex normals of each vertex
    vert_normals.resize(vertices.size());
    for (unsigned int i = 0; i < vert_normals.size(); i++)
        vert_normals[i].resize(3);

    // Loop through each vertex
    for (unsigned int i = 0; i < vertices.size(); i++) {
        
        // Initialise/reset running total of face area of triangles sharing this vertex
        double total_area = 0;

        // Loop through triangle matrix rows
        for (unsigned int j = 0; j < tris.size(); j++) {

            // Loop through triangle matrix columns and find vertex indices
            for (unsigned int k = 0; k < tris[0].size(); k++) {

                // Check if vertex is used by current triangle
                if (i == tris[j][k]) {
                    
                    // Calculate the area of the current triangle using its three vertex points
                    double tri_area = triangle_area(vertices[tris[j][0]], vertices[tris[j][1]], vertices[tris[j][2]]);

                    // Add area-weighted row of face_mat to corresponding vert_normals row
                    vert_normals[i][0] = vert_normals[i][0] + tri_area*face_mat[j][0];
                    vert_normals[i][1] = vert_normals[i][1] + tri_area*face_mat[j][1];
                    vert_normals[i][2] = vert_normals[i][2] + tri_area*face_mat[j][2];
                    
                    // Update running total of face area for this vertex
                    total_area = total_area + tri_area;

                    // Exit innermost loop
                    break;
                }
            }
        }

        // Divide each row of vert_normals by total area of all triangles sharing vertex; weighted average
        vert_normals[i][0] = vert_normals[i][0]/total_area;
        vert_normals[i][1] = vert_normals[i][1]/total_area;
        vert_normals[i][2] = vert_normals[i][2]/total_area;

        // Normalise each row (vector) of vertex normal matrix
        double norm = sqrt(vert_normals[i][0]*vert_normals[i][0] + vert_normals[i][1]*vert_normals[i][1] + vert_normals[i][2]*vert_normals[i][2]);
        vert_normals[i][0] = vert_normals[i][0]/norm;
        vert_normals[i][1] = vert_normals[i][1]/norm;
        vert_normals[i][2] = vert_normals[i][2]/norm;
    }
}

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
    
    // Get vertex_normals
    get_vertex_normals(vertices, tris, vert_normals);   // Invoke function to get vertex normals for "rect"
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
        if (fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,\n", \
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
        if (fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,\n", \
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

    // Get rotated vertex normals
    vert_normals = vertex_rotation(vert_normals, yaw, pitch, roll); // Rotation
}


/* Main RTS function */

namespace rs {

    // Define function RTS for ray-tracing implementation; initially declared in rsworld.cuh
    void RTS(World *world)
    {
    	// Indicate RTS start
    	printf("Setting up RTS...\n");

    	/* Timer for graph setup code: start time */
    	struct timeval timer;
    	gettimeofday(&timer, NULL);
    	double setup_StartTime = timer.tv_sec + (timer.tv_usec/1000000.0);

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
        RTvariable rtvar_width;         // Number of rays; width of virtual screen
        RTvariable rtvar_height;        // Number of rays; height of virtual screen
        RTvariable rtvar_depth;         // Number of rays; depth of virtual screen
        RTvariable rtvar_maxReflDepth;  // Number of allowed ray reflections
        RTvariable rtvar_maxRefrDepth;  // Number of allowed ray refractions
        RTvariable rtvar_rxsize;        // Number of receivers
        RTvariable rtvar_rayOrigin;     // Coordinates of point source location
        RTvariable rtvar_txBoresight;   // Tx boresight spans and launch range
        RTvariable rtvar_txDir;         // Tx boresight direction
        RTvariable rtvar_targRCS;       // Target RCS


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

        RTbuffer    rtbuf_results;                                      // Name of buffer that will contain OptiX outputs
        PerRayData* hbuf_results;       RTvariable rtvar_results;       // Variables for output buffer

        RTbuffer    rtbuf_targ_vel;                                     // Name of buffer to hold target positions
        double3* hbuf_targ_vel;  RTvariable rtvar_targ_vel;             // Variables for target positions array


        // Set up ray count and reflection/refraction depths
        uint3 rts_vars = rsParameters::GetRTSVariables();    
        unsigned int h_numRays = rts_vars.x;            // Number of rays spawned in each dimension (x, y, z) for each target
        unsigned int h_maxReflDepth = rts_vars.y;       // Maximum number of reflections; user input (max. desired reflections per ray)
        unsigned int h_maxRefrDepth = rts_vars.z;       // Maximum number of refractions


        /* Create output buffer */
        unsigned int rayTotal = (h_numRays*h_numRays*h_numRays)*(1 + h_maxRefrDepth);
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
        double sim_endtime = rsParameters::end_time();                      // End time of the simulation
        double sample_time = 1.0 / rsParameters::cw_sample_rate();          // Default CW sample rate is 1 kHz


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
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_height", &rtvar_height ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_height, h_numRays ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_depth", &rtvar_depth ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_depth, h_numRays ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_rayOrigin", &rtvar_rayOrigin ) );     // Value set later; changes over time
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_txBoresight", &rtvar_txBoresight ) ); // Value set later; changes over time
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_txDir", &rtvar_txDir ) );             // Value set later; changes over time
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "d_rxsize", &rtvar_rxsize ) );
        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_rxsize, rxsize ) );

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
        RTvariable      rtvar_targReflCoeff;   // Reflection coefficient
        RTvariable      rtvar_targRefrIndex;    // Refractive index

        /* Declare geometry instance node objects */
        RTgeometryinstance rtnode_geoInst;      // Name of geometry instance


        /* *************** GEOMETRY TOP GROUP NODE SETUP *************** */

        // Declare top-level structures for all target geometry
        RTtransform     transforms[targsize];
        RTgroup         rtTop_group;
        RTvariable      rtTop_object;
        RTacceleration  rtTop_acceleration;
        float affine_matrix[16] = { 1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 1, 0,
                                    0, 0, 0, 1 };  // Used to apply transformation to target objects in RTS

        // Declare context variables for all targets
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "rtTop_object", &rtTop_object ) );
        RT_CHECK_ERROR( rtContextDeclareVariable( context, "dbuf_targ_vel", &rtvar_targ_vel ) );


        // Finish time for setting up RTS before time-steps loop
        gettimeofday(&timer, NULL);
        double setup_FinishTime = timer.tv_sec + (timer.tv_usec/1000000.0);


        /* *************** START OF TRANSMITTER LOOP *************** */

        // Get current time when starting time-step loops
        gettimeofday(&timer, NULL);
        double timeloop_StartTime = timer.tv_sec + (timer.tv_usec/1000000.0);

        // Iterate through all transmitters
        for (unsigned int tx_i = 0; tx_i < txsize; tx_i++){

            // Transmitter and signal set-up
            Transmitter* trans = trans_arr[tx_i];                               // Each transmitter in the loop
            unsigned int pulseCount = trans->GetPulseCount();                   // Number of pulses to transmit
            TransmitterPulse* signal = new TransmitterPulse();                  // Create new signal
            trans->GetPulse(signal, 0);                                         // Define pulse signal
            RadarSignal *wave = signal->wave;                                   // Pulse wave
            double pulse_length = wave->GetLength();                            // Pulse length
            double carrier = wave->GetCarrier();                                // Carrier frequency
            double Wl = cspeed/carrier;                                         // Wavelength
            double3 h_rayOrigin;                                                // Initialise transmitter coordinates
            double2 h_txDir;                                                    // Initialise transmitter boresight direction
            double3 h_txBoresight = trans->GetTxBoresight();                    // Boresight spans and launch range
            RT_CHECK_ERROR( rtVariableSetUserData(rtvar_txBoresight, sizeof(double3), &h_txBoresight) );

            // Time set-up
            unsigned int point_count = ceil(pulse_length / sample_time);        // Number of interpolation points we need to add
            unsigned int numPoints = point_count + 1;                           // Number of Points
            vector < double > start_time_arr(pulseCount);                       // start_time vector varying with pulse number
            
            // Create vector of PRDs; holds the PRD results for ALL rays over ALL time instances; resets for EACH transmitter
            vector < PerRayData* > prd_vec(pulseCount*numPoints);               // Requires total number of time instances to simulate


            /// Assign receiver buffer values; these values do NOT change over time, so no need to repeat these later in the time-step loops
            RT_CHECK_ERROR( rtBufferMap(rtbuf_sphRadius, (void **)&hbuf_sphRadius) );   // Map sphRadius buffer
            for (unsigned j = 0; j < rxsize; j++) {

                // Set overall noise temperature for receiver (antenna + external noise); recorded later into responses
                recv_arr[j]->SetNoiseTemperature(wave->GetTemp() + recv_arr[j]->GetNoiseTemperature());

                // Get radius and theta and phi spans
                double3 rxsphere = recv_arr[j]->GetRxSphere();                          // x = radius; y = thetaSpan; z = phiSpan
                hbuf_sphRadius[j] = rxsphere.x;
                hbuf_sphRadius[j] = 0.30;                                               // CHANGE: Remove when ready to use correct values
            }

            // Unmap receiver buffers
            RT_CHECK_ERROR( rtBufferUnmap(rtbuf_sphRadius) );                           // Unmap buffer



            /* *************** START OF TIME-STEP LOOPS *************** */

            // Iterate through all pulses
            for (unsigned int k = 0; k < 1; k++){ // CHANGE BACK TO: pulseCount; k++){

                // Pulse start time
                trans->GetPulse(signal, k);                                                             // Pulse signal
                start_time_arr[k] = signal->time;

                // Iterate through all Points
                for (unsigned int n = 0; n < numPoints; n++){

                    // Set up time variables
                    double time_t;
                    if (n == point_count)
                        time_t = start_time_arr[k] + pulse_length;                                      // Final Point
                    else
                        time_t = start_time_arr[k] + (n*sample_time);                                   // Time at the start of the sample

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
                    h_rayOrigin.x = 0;                      // Test Tx coordinates; CHANGE: Remove when ready to use correct values below
                    h_rayOrigin.y = 0;                      // Test Tx coordinates; CHANGE: Remove when ready to use correct values below
                    h_rayOrigin.z = 200;                    // Test Tx coordinates; CHANGE: Remove when ready to use correct values below
                    RT_CHECK_ERROR( rtVariableSetUserData(rtvar_rayOrigin, sizeof(double3), &h_rayOrigin) );

                    // Set up initial Tx boresight and beamwidth; used to direct the spawned rays
                    h_txDir.x = (trans->GetRotation(time_t)).azimuth;
                    h_txDir.y = (trans->GetRotation(time_t)).elevation;
                    RT_CHECK_ERROR( rtVariableSetUserData(rtvar_txDir, sizeof(double2), &h_txDir) );


                    /// Assign receiver buffer values; these values change over time
                    for (unsigned j = 0; j < rxsize; j++) {

                        // Rotate Rx sphere centre onto Rx boresight vector; use spherical coordinates of sphere centre relative to Rx coordinates as the origin
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

                        // CHANGE: Remove this block of code when ready to use actual Rx sphere centre values
                        hbuf_sphCentre[j].x = 0;
                        hbuf_sphCentre[j].y = 0;
                        hbuf_sphCentre[j].z = 0;
                        h_Rx_azimuth = M_PI/2;      // Azimuth from sphere centre to receiver
                        h_Rx_elevation = M_PI/2;    // Elevation from sphere centre to receiver
                        rxsphere.y = 2;             // Theta span
                        rxsphere.z = M_PI/2;        // Phi span

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

                        // Get and save target centre positions; CHANGE: Uncomment and remove *100 stuff in .z
                        targ_positions[targ_i].x = 0; //targ_arr[targ_i]->GetPosition(time_t).x;
                        targ_positions[targ_i].y = 0; //targ_arr[targ_i]->GetPosition(time_t).y;
                        targ_positions[targ_i].z = targ_i*100; //targ_arr[targ_i]->GetPosition(time_t).z;

                        // Get and save "next" target centre positions (one sample later); used for Doppler calculation; CHANGE: Uncomment and remove *100 stuff in .z
                        targ_positions_end[targ_i].x = 0; //targ_arr[targ_i]->GetPosition(time_t + sample_time).x;
                        targ_positions_end[targ_i].y = 0; //targ_arr[targ_i]->GetPosition(time_t + sample_time).y;
                        targ_positions_end[targ_i].z = targ_i*100; //targ_arr[targ_i]->GetPosition(time_t + sample_time).z;

                		// Create target vertex matrices
                        vector < vector < unsigned int > > tris;    // Target's triangles matrix
                        vector < vector < double > > verts;         // Target's vertices matrix
                        vector < vector < double > > vert_normals;  // Target's vertex normals matrix

                        // Get the initial rotation/orientation of the target (at time t = 0)
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
                            
                        // CHANGE: Uncomment the following when ready to use actual target coordinates
                        // for (unsigned int i = 0; i < verts.size(); i++) {
                        //     verts[i][0] += targ_positions[targ_i].x;
                        //     verts[i][1] += targ_positions[targ_i].y;
                        //     verts[i][2] += targ_positions[targ_i].z;
                        // }


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
                        double targRCS = targ_arr[targ_i]->GetRCS();
                        RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targIndex", &rtvar_targ_index ) );            // Declare targ_index
                        RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targReflCoeff", &rtvar_targReflCoeff ) );     // Declare reflCoeff
                        RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targRefrIndex", &rtvar_targRefrIndex ) );     // Declare refrIndex
                        RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "d_targRCS", &rtvar_targRCS ) );                 // Declare targRCS
                        RT_CHECK_ERROR( rtVariableSet1ui( rtvar_targ_index, targ_i ) );                                                     // Set targ_index
                        RT_CHECK_ERROR( rtVariableSetUserData( rtvar_targReflCoeff, sizeof(double), &targReflCoeff ) );                     // Set reflCoeff
                        RT_CHECK_ERROR( rtVariableSetUserData( rtvar_targRefrIndex, sizeof(double), &targRefrIndex ) );                     // Set refrIndex
                        RT_CHECK_ERROR( rtVariableSetUserData( rtvar_targRCS, sizeof(double), &targRCS ) );                                 // Set targRCS


                        /* *************** GEOMETRY GROUP NODE SETUP *************** */

                        /* Create geometry group node */
                        RT_CHECK_ERROR( rtGeometryGroupCreate( context, &rtnode_geoGroup ) );
                        RT_CHECK_ERROR( rtGeometryGroupSetChildCount( rtnode_geoGroup, 1 ) );
                        RT_CHECK_ERROR( rtGeometryGroupSetChild( rtnode_geoGroup, 0, rtnode_geoInst ) );

                        /* Specify an acceleration structure */
                        RT_CHECK_ERROR( rtAccelerationCreate( context, &rtnode_geoGroupAcc ) );
                        RT_CHECK_ERROR( rtAccelerationSetBuilder( rtnode_geoGroupAcc, "Bvh" ) );
                        RT_CHECK_ERROR( rtAccelerationSetTraverser( rtnode_geoGroupAcc, "Bvh" ) );
                        RT_CHECK_ERROR( rtGeometryGroupSetAcceleration( rtnode_geoGroup, rtnode_geoGroupAcc) );
                        RT_CHECK_ERROR( rtAccelerationMarkDirty( rtnode_geoGroupAcc ) );

                        /* Add a transform node */
                        RT_CHECK_ERROR( rtTransformCreate( context, &transforms[targ_i] ) );
                        RT_CHECK_ERROR( rtTransformSetChild( transforms[targ_i], rtnode_geoGroup ) );
                        // CHANGE: Uncomment the following when ready to use actual target coordinates
                        // affine_matrix[3] = targ_positions[targ_i].x;    // Translate the target to its centre coordinates (determined by SGP4)
                        // affine_matrix[7] = targ_positions[targ_i].y;
                        // affine_matrix[11] = targ_positions[targ_i].z;
                        RT_CHECK_ERROR( rtTransformSetMatrix( transforms[targ_i], 0, affine_matrix, 0 ) );  // Apply transform


                        /* *************** TARGET MATRICES SETUP *************** */
                        
                        /* Create and fill triangle buffer; not required for file mesh */
                        RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( rtnode_geoInst, "dbuf_triangles", &rtvar_triangles ) );  // Declare device variable dbuf_triVertices
                        RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_triangles ) );                 // Create buffer
                        RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_triangles, RT_FORMAT_UNSIGNED_INT3 ) );                // Set buffer to hold double3s
                        RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_triangles, tris.size() ) );                            // Set size of buffer
                        RT_CHECK_ERROR( rtVariableSetObject( rtvar_triangles, rtbuf_triangles ) );                      // Associate contents
                        RT_CHECK_ERROR( rtBufferMap(rtbuf_triangles, (void **)&hbuf_triangles) );                       // Map buffer

                        // Populate buffer with verts vector
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
                    }


                    /* *************** END OF TARGET LOOP *************** */


                    /* Place GeometryGroups as children of the top-level object */
                    RT_CHECK_ERROR( rtGroupCreate( context, &rtTop_group ) );
                    RT_CHECK_ERROR( rtGroupSetChildCount( rtTop_group, targsize ) );
                    for (unsigned int targ_i = 0; targ_i < targsize; targ_i++)
                        RT_CHECK_ERROR( rtGroupSetChild( rtTop_group, targ_i, transforms[targ_i] ) );

                    RT_CHECK_ERROR( rtVariableSetObject( rtTop_object, rtTop_group ) );
                    RT_CHECK_ERROR( rtAccelerationCreate( context, &rtTop_acceleration ) );
                    RT_CHECK_ERROR( rtAccelerationSetBuilder(rtTop_acceleration, "Bvh") );
                    RT_CHECK_ERROR( rtAccelerationSetTraverser(rtTop_acceleration, "Bvh") );
                    RT_CHECK_ERROR( rtGroupSetAcceleration( rtTop_group, rtTop_acceleration) );
                    RT_CHECK_ERROR( rtAccelerationMarkDirty( rtTop_acceleration ) );


                    /* Create and fill target positions buffer - all targets' positions at ONE time instance */
                    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &rtbuf_targ_vel ) );              // Create buffer
                    RT_CHECK_ERROR( rtBufferSetFormat( rtbuf_targ_vel, RT_FORMAT_USER ) );                      // Set buffer to hold double3s
                    RT_CHECK_ERROR( rtBufferSetElementSize( rtbuf_targ_vel, sizeof(double3) ) );                // Use element size of double3
                    RT_CHECK_ERROR( rtBufferSetSize1D( rtbuf_targ_vel, targsize ) );                            // Set size of buffer
                    RT_CHECK_ERROR( rtVariableSetObject( rtvar_targ_vel, rtbuf_targ_vel ) );                    // Associate contents
                    RT_CHECK_ERROR( rtBufferMap(rtbuf_targ_vel, (void **)&hbuf_targ_vel) );                     // Map buffer

                    // Populate buffer with the 2-D targ_positions vector
                    for (unsigned int i = 0; i < targsize; i++)
                        hbuf_targ_vel[i] = (targ_positions_end[i] - targ_positions[i])/sample_time;             // Velocity over one sample
                    RT_CHECK_ERROR( rtBufferUnmap(rtbuf_targ_vel) );                                            // Unmap buffer


                    /* *************** LAUNCH OPTIX KERNEL *************** */
                	
                 	/* Validate and compile OptiX */
                	RT_CHECK_ERROR( rtContextValidate( context ) );
                	RT_CHECK_ERROR( rtContextCompile( context ) );

                  	/* Launch OptiX */
                	RT_CHECK_ERROR( rtContextLaunch3D( context, 0 /* entry point */, h_numRays, h_numRays, h_numRays ) );

                    // Map OptiX result buffer
                    RT_CHECK_ERROR( rtBufferMap(rtbuf_results, (void **)&hbuf_results) );
                    
                    // Loop through all PRDs
                    for (unsigned int i = 0; i < rayTotal; i++)
                    {
                        if (hbuf_results[i].received >= 0)   // If ray has hit the receiver
                        {
                            // Set up transvec and recvvec
                            SVec3 transvec, recvvec;

                            // Calculate the value of Pr (exlcuding Pt) for each ray
                            if ((hbuf_results[i].reflDepth == 0) && (hbuf_results[i].refrDepth == 0)) { // If direct transmission
                                transvec = SVec3(d3_to_V3(h_rayOrigin - hbuf_results[i].endPoint));
                                recvvec = SVec3(d3_to_V3(hbuf_results[i].endPoint - h_rayOrigin));
                            }
                            else {                                                                      // If indirect transmission
                                transvec = SVec3(d3_to_V3(hbuf_results[i].firstHitPoint - h_rayOrigin));
                                recvvec = SVec3(d3_to_V3(hbuf_results[i].endPoint - hbuf_results[i].prevHitPoint));
                            }

                            // Calculate gains and update power
                            transvec.length = 1;                                                        // Normalise
                            recvvec.length = 1;                                                         // Normalise
                            double delay = (hbuf_results[i].rayLength)/cspeed;                          // (Distance travelled by ray) / c
                            double Gt = trans->GetGain(transvec, trans->GetRotation(time_t), Wl);
                            double Gr = (world->receivers)[hbuf_results[i].received]->GetGain(recvvec, \
                                            (world->receivers)[hbuf_results[i].received]->GetRotation(delay + time_t), Wl);
                            hbuf_results[i].power *= (Wl*Wl*Gt*Gr);                                     // Power is multiplied by Pt in "rsresponse"
                            
                            // Update Doppler; currently holds total Doppler velocity = 2Vr; to be divided by Wl (for Fd = 2Vr/Wl)
                            double Vr = hbuf_results[i].doppler/2;                                      // First divide by 2 to get overall Vr
                            hbuf_results[i].doppler = carrier*(((1 + Vr/cspeed)/(1 - Vr/cspeed)) - 1);  // Insert Vr into POMR equation, then Fd = Fr - Fc
                        }
                    }    

                    // Save current results buffer in prd_vec at time_t
                    prd_vec[n + (numPoints*k)] = hbuf_results;

                    // Done with results buffer; unmap buffer
                    RT_CHECK_ERROR( rtBufferUnmap(rtbuf_results) );
                }


                /* *************** ADD RESPONSES TO RECEIVER *************** */
                
                // Loop through rays; must be done AFTER Points loop
                for (unsigned int i = 0; i < rayTotal; i++) {

                    // New response for every received ray and every pulse
                    Response *response = new Response(wave, trans);

                    // Loop through Points
                    for (unsigned int n = 0; n < point_count; n++) {

                        // Access the appropriate results buffer from prd_vec
                        PerRayData* results = prd_vec[n + (numPoints*k)];

                        // If ray has hit the receiver, add it to the response
                        if (results[i].received >= 0) {

                            // Calculate the delay (in seconds) experienced by the pulse; only needs rayLength from PRD
                            double delay = (results[i].rayLength)/cspeed;                               // (Distance travelled by ray) / c

                            // Calculate phase shift; see "Phase Delay Equation" in doc/equations/equations.tex; needs nothing from PRD
                            double phase = -delay*2*M_PI*carrier;                                       // Power is multiplied by Pt later (in "rsresponse")

                            // Only add the Point if it will be received within the simulation time
                            if (((n * sample_time) + start_time_arr[k] + delay) < sim_endtime){
                                InterpPoint point(results[i].power, (n * sample_time) + start_time_arr[k] + delay, delay, \
                                                    results[i].doppler, phase, (world->receivers)[results[i].received]->GetNoiseTemperature(), i);
                                response->AddInterpPoint(point);                                        // Add the point to the response
                            }
                        }
                    }

                    // Add the final Point
                    PerRayData* results = prd_vec[point_count + (numPoints*k)];                         // Include the final Point

                    // If ray has hit the receiver, add it to the response
                    if (results[i].received >= 0) {
                        double delay = (results[i].rayLength)/cspeed;                                   // (Distance travelled by ray) / c
                        double phase = -delay*2*M_PI*carrier;

                        // Only add the Point if it will be received within the simulation time
                        if ((start_time_arr[k] + pulse_length + delay) < sim_endtime){
                            InterpPoint point(results[i].power, start_time_arr[k] + pulse_length + delay, delay, \
                                              results[i].doppler, phase, (world->receivers)[results[i].received]->GetNoiseTemperature(), i);
                            response->AddInterpPoint(point);                                            // Add the final Point to the response
                        }
                    }

                    // Add the overall response to the receiver (if there are Points recorded in the response)
                    if (response->CountPoints() > 0)
                        (world->receivers)[results[i].received]->AddResponse(response);        // Add to correct receiver
                }
            }


            /* *************** END OF TIME-STEP LOOPS AND TIMING *************** */
        
        }

        /* Timer for setup code: overall loop time */
        gettimeofday(&timer, NULL);
        double loop_Time = timer.tv_sec + (timer.tv_usec/1000000.0) - timeloop_StartTime;


        /* *************** END OF TRANSMITTERS LOOP *************** */


        /* *************** PRINT TIMING STATS *************** */

        printf("Input and set-up took %lf seconds.\n", setup_FinishTime - setup_StartTime);
        printf("OptiX main loop took %lf seconds.\n", loop_Time);

             	
        /* *************** FREE MEMORY *************** */

        RT_CHECK_ERROR( rtProgramDestroy( rtprog_ray_generation ) );
    	RT_CHECK_ERROR( rtProgramDestroy( rtprog_miss ) );
        // RT_CHECK_ERROR( rtProgramDestroy( rtprog_exception ) );
        RT_CHECK_ERROR( rtProgramDestroy( rtprog_closest_hit ) );
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
