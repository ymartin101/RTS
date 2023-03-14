
/* ****************** HEADER FILE ******************

 * Scene epsilons and PRD initialisation

 ************************************************ */

// Set variables required by RTS
#define SCENE_EPS 0.005f	// Default scene epsilon for incident and refracted rays (needs to allow small rays); minimum incident ray length
#define SCENE_EPS_R 0.005f	// Default scene epsilon for reflected rays; minimum reflected ray length

// PRD structure - tracks the ray information for each ray; returned to host after ray traversal
struct PerRayData
{
	double rayLength;			// Ray's total pathlength
	double2 refrIndex;			// Ray's previous and current refraction indices; used to calculate ratio of indices needed for refraction (n2/n1)
	unsigned int reflDepth;		// Number of reflections
	unsigned int refrDepth;		// Number of refractions
	unsigned int maxRayIndex;	// Maximum ray index up to this point (accounts for refractions)
	double3 rayDirection;		// Unit vector of ray direction
	double3 firstHitPoint;		// Coordinates of ray's first hit point
	double3 prevHitPoint;		// Coordinates of ray's previous hit point; initially used as ray origin until reflection/refraction
	// double3 endPoint;			// Coordinates of ray's end point
	double power;				// Power of ray; changes at each target, then at Rx, and then finalised AFTER ray traversal
	double doppler;				// Doppler frequency of ray
	int received;				// Receive status of ray; -1 for NOT RECEIVED; otherwise, it is the receiver index
	bool end;					// Should the ray be terminated?
};
