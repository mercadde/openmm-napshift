//TODO: actually include openmm/platforms/cuda/src/kernels/vectorOps.cu
/**
 * This file defines vector operations to simplify code elsewhere.
 */

// Versions of make_x() that take a single value and set all components to that.

inline __device__ int2 make_int2(int a) {
    return make_int2(a, a);
}

inline __device__ int3 make_int3(int a) {
    return make_int3(a, a, a);
}

inline __device__ int4 make_int4(int a) {
    return make_int4(a, a, a, a);
}

inline __device__ float2 make_float2(float a) {
    return make_float2(a, a);
}

inline __device__ float3 make_float3(float a) {
    return make_float3(a, a, a);
}

inline __device__ float4 make_float4(float a) {
    return make_float4(a, a, a, a);
}

inline __device__ double2 make_double2(double a) {
    return make_double2(a, a);
}

inline __device__ double3 make_double3(double a) {
    return make_double3(a, a, a);
}

inline __device__ double4 make_double4(double a) {
    return make_double4(a, a, a, a);
}

// Negate a vector.

inline __device__ int2 operator-(int2 a) {
    return make_int2(-a.x, -a.y);
}

inline __device__ int3 operator-(int3 a) {
    return make_int3(-a.x, -a.y, -a.z);
}

inline __device__ int4 operator-(int4 a) {
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ float2 operator-(float2 a) {
    return make_float2(-a.x, -a.y);
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ float4 operator-(float4 a) {
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ double2 operator-(double2 a) {
    return make_double2(-a.x, -a.y);
}

inline __device__ double3 operator-(double3 a) {
    return make_double3(-a.x, -a.y, -a.z);
}

inline __device__ double4 operator-(double4 a) {
    return make_double4(-a.x, -a.y, -a.z, -a.w);
}

// Add two vectors.

inline __device__ int2 operator+(int2 a, int2 b) {
    return make_int2(a.x+b.x, a.y+b.y);
}

inline __device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ int4 operator+(int4 a, int4 b) {
    return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x+b.x, a.y+b.y);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline __device__ double2 operator+(double2 a, double2 b) {
    return make_double2(a.x+b.x, a.y+b.y);
}

inline __device__ double3 operator+(double3 a, double3 b) {
    return make_double3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ double4 operator+(double4 a, double4 b) {
    return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

// Subtract two vectors.

inline __device__ int2 operator-(int2 a, int2 b) {
    return make_int2(a.x-b.x, a.y-b.y);
}

inline __device__ int3 operator-(int3 a, int3 b) {
    return make_int3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ int4 operator-(int4 a, int4 b) {
    return make_int4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x-b.x, a.y-b.y);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

inline __device__ double2 operator-(double2 a, double2 b) {
    return make_double2(a.x-b.x, a.y-b.y);
}

inline __device__ double3 operator-(double3 a, double3 b) {
    return make_double3(a.x-b.x, a.y-b.y, a.z-b.z);
}

inline __device__ double4 operator-(double4 a, double4 b) {
    return make_double4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

// Multiply two vectors.

inline __device__ int2 operator*(int2 a, int2 b) {
    return make_int2(a.x*b.x, a.y*b.y);
}

inline __device__ int3 operator*(int3 a, int3 b) {
    return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ int4 operator*(int4 a, int4 b) {
    return make_int4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x*b.x, a.y*b.y);
}

inline __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ float4 operator*(float4 a, float4 b) {
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

inline __device__ double2 operator*(double2 a, double2 b) {
    return make_double2(a.x*b.x, a.y*b.y);
}

inline __device__ double3 operator*(double3 a, double3 b) {
    return make_double3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ double4 operator*(double4 a, double4 b) {
    return make_double4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}

// Divide two vectors.

inline __device__ int2 operator/(int2 a, int2 b) {
    return make_int2(a.x/b.x, a.y/b.y);
}

inline __device__ int3 operator/(int3 a, int3 b) {
    return make_int3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ int4 operator/(int4 a, int4 b) {
    return make_int4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

inline __device__ float2 operator/(float2 a, float2 b) {
    return make_float2(a.x/b.x, a.y/b.y);
}

inline __device__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

inline __device__ double2 operator/(double2 a, double2 b) {
    return make_double2(a.x/b.x, a.y/b.y);
}

inline __device__ double3 operator/(double3 a, double3 b) {
    return make_double3(a.x/b.x, a.y/b.y, a.z/b.z);
}

inline __device__ double4 operator/(double4 a, double4 b) {
    return make_double4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}

// += operator

inline __device__ void operator+=(int2& a, int2 b) {
    a.x += b.x; a.y += b.y;
}

inline __device__ void operator+=(int3& a, int3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator+=(int4& a, int4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __device__ void operator+=(float2& a, float2 b) {
    a.x += b.x; a.y += b.y;
}

inline __device__ void operator+=(float3& a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator+=(float4& a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __device__ void operator+=(double2& a, double2 b) {
    a.x += b.x; a.y += b.y;
}

inline __device__ void operator+=(double3& a, double3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __device__ void operator+=(double4& a, double4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// -= operator

inline __device__ void operator-=(int2& a, int2 b) {
    a.x -= b.x; a.y -= b.y;
}

inline __device__ void operator-=(int3& a, int3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator-=(int4& a, int4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __device__ void operator-=(float2& a, float2 b) {
    a.x -= b.x; a.y -= b.y;
}

inline __device__ void operator-=(float3& a, float3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator-=(float4& a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __device__ void operator-=(double2& a, double2 b) {
    a.x -= b.x; a.y -= b.y;
}

inline __device__ void operator-=(double3& a, double3 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator-=(double4& a, double4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// *= operator

inline __device__ void operator*=(int2& a, int2 b) {
    a.x *= b.x; a.y *= b.y;
}

inline __device__ void operator*=(int3& a, int3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator*=(int4& a, int4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

inline __device__ void operator*=(float2& a, float2 b) {
    a.x *= b.x; a.y *= b.y;
}

inline __device__ void operator*=(float3& a, float3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator*=(float4& a, float4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

inline __device__ void operator*=(double2& a, double2 b) {
    a.x *= b.x; a.y *= b.y;
}

inline __device__ void operator*=(double3& a, double3 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ void operator*=(double4& a, double4 b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
}

// /= operator

inline __device__ void operator/=(int2& a, int2 b) {
    a.x /= b.x; a.y /= b.y;
}

inline __device__ void operator/=(int3& a, int3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}

inline __device__ void operator/=(int4& a, int4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

inline __device__ void operator/=(float2& a, float2 b) {
    a.x /= b.x; a.y /= b.y;
}

inline __device__ void operator/=(float3& a, float3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}

inline __device__ void operator/=(float4& a, float4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

inline __device__ void operator/=(double2& a, double2 b) {
    a.x /= b.x; a.y /= b.y;
}

inline __device__ void operator/=(double3& a, double3 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z;
}

inline __device__ void operator/=(double4& a, double4 b) {
    a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;
}

// Multiply a vector by a constant.

inline __device__ int2 operator*(int2 a, int b) {
    return make_int2(a.x*b, a.y*b);
}

inline __device__ int3 operator*(int3 a, int b) {
    return make_int3(a.x*b, a.y*b, a.z*b);
}

inline __device__ int4 operator*(int4 a, int b) {
    return make_int4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ int2 operator*(int a, int2 b) {
    return make_int2(a*b.x, a*b.y);
}

inline __device__ int3 operator*(int a, int3 b) {
    return make_int3(a*b.x, a*b.y, a*b.z);
}

inline __device__ int4 operator*(int a, int4 b) {
    return make_int4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ float2 operator*(float2 a, float b) {
    return make_float2(a.x*b, a.y*b);
}

inline __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ float2 operator*(float a, float2 b) {
    return make_float2(a*b.x, a*b.y);
}

inline __device__ float3 operator*(float a, float3 b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __device__ float4 operator*(float a, float4 b) {
    return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

inline __device__ double2 operator*(double2 a, double b) {
    return make_double2(a.x*b, a.y*b);
}

inline __device__ double3 operator*(double3 a, double b) {
    return make_double3(a.x*b, a.y*b, a.z*b);
}

inline __device__ double4 operator*(double4 a, double b) {
    return make_double4(a.x*b, a.y*b, a.z*b, a.w*b);
}

inline __device__ double2 operator*(double a, double2 b) {
    return make_double2(a*b.x, a*b.y);
}

inline __device__ double3 operator*(double a, double3 b) {
    return make_double3(a*b.x, a*b.y, a*b.z);
}

inline __device__ double4 operator*(double a, double4 b) {
    return make_double4(a*b.x, a*b.y, a*b.z, a*b.w);
}

// Divide a vector by a constant.

inline __device__ int2 operator/(int2 a, int b) {
    return make_int2(a.x/b, a.y/b);
}

inline __device__ int3 operator/(int3 a, int b) {
    return make_int3(a.x/b, a.y/b, a.z/b);
}

inline __device__ int4 operator/(int4 a, int b) {
    return make_int4(a.x/b, a.y/b, a.z/b, a.w/b);
}

inline __device__ float2 operator/(float2 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ float3 operator/(float3 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ float4 operator/(float4 a, float b) {
    float scale = 1.0f/b;
    return a*scale;
}

inline __device__ double2 operator/(double2 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

inline __device__ double3 operator/(double3 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

inline __device__ double4 operator/(double4 a, double b) {
    double scale = 1.0/b;
    return a*scale;
}

// *= operator (multiply vector by constant)

inline __device__ void operator*=(int2& a, int b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(int3& a, int b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(int4& a, int b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __device__ void operator*=(float2& a, float b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(float3& a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(float4& a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

inline __device__ void operator*=(double2& a, double b) {
    a.x *= b; a.y *= b;
}

inline __device__ void operator*=(double3& a, double b) {
    a.x *= b; a.y *= b; a.z *= b;
}

inline __device__ void operator*=(double4& a, double b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
}

// Dot product

inline __device__ float dot(float3 a, float3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline __device__ double dot(double3 a, double3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

// Cross product

inline __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline __device__ float4 cross(float4 a, float4 b) {
    return make_float4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0f);
}

inline __device__ double3 cross(double3 a, double3 b) {
    return make_double3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

inline __device__ double4 cross(double4 a, double4 b) {
    return make_double4(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x, 0.0);
}

// Normalize a vector

inline __device__ float2 normalize(float2 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y);
}

inline __device__ float3 normalize(float3 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
}

inline __device__ float4 normalize(float4 a) {
    return a*rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}

inline __device__ double2 normalize(double2 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y);
}

inline __device__ double3 normalize(double3 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}

inline __device__ double4 normalize(double4 a) {
    return a*rsqrt(a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w);
}

extern "C" __device__
real calculateDihedral(int a1, int a2, int a3, int a4, real3* forceOutputs, const int* __restrict__ indexToAtom, const real4* __restrict__ posq, real4 periodicBoxSize, real4 invPeriodicBoxSize) {
    if ( a1 == -1 || a2 == -1 || a3 == -1 || a4 == -1 ) {
        return -100;
    }
    real4 pos1 = posq[indexToAtom[a1]];
    real4 pos2 = posq[indexToAtom[a2]];
    real4 pos3 = posq[indexToAtom[a3]];
    real4 pos4 = posq[indexToAtom[a4]];

    const real PI = (real) 3.14159265358979323846;
    real3 v0 = make_real3(pos1.x-pos2.x, pos1.y-pos2.y, pos1.z-pos2.z);
    real3 v1 = make_real3(pos3.x-pos2.x, pos3.y-pos2.y, pos3.z-pos2.z);
    real3 v2 = make_real3(pos3.x-pos4.x, pos3.y-pos4.y, pos3.z-pos4.z);
    #if APPLY_PERIODIC
    APPLY_PERIODIC_TO_DELTA(v0)
    APPLY_PERIODIC_TO_DELTA(v1)
    APPLY_PERIODIC_TO_DELTA(v2)
    #endif
    real3 cp0 = cross(v0, v1);
    real3 cp1 = cross(v1, v2);
    real cosangle = dot(normalize(cp0), normalize(cp1));
    real theta;
    if (cosangle > 0.99f || cosangle < -0.99f) {
        real3 cross_prod = cross(cp0, cp1);
        real scale = dot(cp0, cp0)*dot(cp1, cp1);
        theta = ASIN(SQRT(dot(cross_prod, cross_prod)/scale));
        if (cosangle < 0)
            theta = PI-theta;
    } else {
        theta = ACOS(cosangle);
    }
    theta = (dot(v0, cp1) >= 0 ? theta : -theta);
    
    real normCross1 = dot(cp0, cp0);
    real normSqrBC = dot(v1, v1);
    real normBC = SQRT(normSqrBC);
    real normCross2 = dot(cp1, cp1);
    real dp = RECIP(normSqrBC);
    real4 ff = make_real4((-normBC)/normCross1, dot(v0, v1)*dp, dot(v2, v1)*dp, (normBC)/normCross2);
    
    real3 force1 = ff.x*cp0;
    real3 force4 = ff.w*cp1;
    real3 s = ff.y*force1 - ff.z*force4;
    real3 force2 = s-force1;
    real3 force3 = -s-force4;

    forceOutputs[0] = force1;
    forceOutputs[1] = force2;
    forceOutputs[2] = force3;
    forceOutputs[3] = force4;

    return theta;
}

extern "C" __device__
real calculateAngle(int a1, int a2, int a3, real3* forceOutputs, const int* __restrict__ indexToAtom, const real4* __restrict__ posq, real4 periodicBoxSize, real4 invPeriodicBoxSize) {
    if ( a1 == -1 || a2 == -1 || a3 == -1) {
        return -100;
    }

    real4 pos1 = posq[indexToAtom[a1]];
    real4 pos2 = posq[indexToAtom[a2]];
    real4 pos3 = posq[indexToAtom[a3]];

    real3 v0 = make_real3(pos2.x-pos1.x, pos2.y-pos1.y, pos2.z-pos1.z);
    real3 v1 = make_real3(pos2.x-pos3.x, pos2.y-pos3.y, pos2.z-pos3.z);
    
    #if APPLY_PERIODIC
    APPLY_PERIODIC_TO_DELTA(v0)
    APPLY_PERIODIC_TO_DELTA(v1)
    #endif
    real3 cp = cross(v0, v1);
    real rp = cp.x*cp.x + cp.y*cp.y + cp.z*cp.z;
    rp = max(SQRT(rp), (real) 1.0e-06f);
    real r21 = v0.x*v0.x + v0.y*v0.y + v0.z*v0.z;
    real r23 = v1.x*v1.x + v1.y*v1.y + v1.z*v1.z;
    real dot = v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
    real cosine = min(max(dot*RSQRT(r21*r23), (real) -1), (real) 1);
    real theta = ACOS(cosine);

    real3 force1 = cross(v0, cp)*(1/(r21*rp));
    real3 force3 = cross(cp, v1)*(1/(r23*rp));
    real3 force2 = -force1-force3;
    
    forceOutputs[0] = force1;
    forceOutputs[1] = force2;
    forceOutputs[2] = force3;

    return theta;
}

extern "C" __global__
void resolveAngle(real* inputTensor,
                  real* __restrict__ angleForceArray,
                  const int* particlesIndices,
                  const int* indexToAtom,
                  const real4* posq,
                  const int numPeptides,
                  const int numAnglesPerPeptide,
                  real4 periodicBoxSize,
                  real4 invPeriodicBoxSize){
                    
    for (int angle_idx = blockIdx.x*blockDim.x+threadIdx.x; angle_idx < numPeptides*numAnglesPerPeptide; angle_idx += blockDim.x*gridDim.x) { 

        int peptide = floor(float(angle_idx) / float(numAnglesPerPeptide));
        int angleNum = angle_idx - peptide*numAnglesPerPeptide;
        real angle = 0;
        real3 angle_force[4] = {0,0,0,0};
        if (angleNum < 2 || numAnglesPerPeptide == 4){ //always calculate diheds in all-atom format
            angle = calculateDihedral(particlesIndices[4*angle_idx+0],
                              particlesIndices[4*angle_idx+1],
                              particlesIndices[4*angle_idx+2],
                              particlesIndices[4*angle_idx+3],
                              angle_force, indexToAtom, posq, periodicBoxSize, invPeriodicBoxSize);
        } else {
            angle = calculateAngle(particlesIndices[4*angle_idx+0],
                              particlesIndices[4*angle_idx+1],
                              particlesIndices[4*angle_idx+2],
                              angle_force, indexToAtom, posq, periodicBoxSize, invPeriodicBoxSize);
        }
        real sinAngle = angle == -100 ? 0 : SIN(angle);
        real cosAngle = angle == -100 ? 0 : COS(angle);

        int idx_left = 3*(2*numAnglesPerPeptide+22)*(peptide-1)+66+4*numAnglesPerPeptide+2*angleNum;
        int idx_center = 3*(2*numAnglesPerPeptide+22)*(peptide)+44+2*numAnglesPerPeptide+2*angleNum;
        int idx_right = 3*(2*numAnglesPerPeptide+22)*(peptide+1)+22+2*angleNum;
        
        //trying to set all three entries for this peptide at once precludes contiguous access... what would be faster?
        if (peptide > 0) {
            inputTensor[idx_left] = sinAngle;
            inputTensor[idx_left+1] = cosAngle;
        }
        inputTensor[idx_center] = sinAngle;
        inputTensor[idx_center+1] = cosAngle;
        if (peptide < numPeptides-1) {
            inputTensor[idx_right] = sinAngle;
            inputTensor[idx_right+1] = cosAngle;
        }

        int sin_force_idx_left = 3*24*numAnglesPerPeptide*(peptide-1)+2*24*numAnglesPerPeptide+12*2*angleNum;
        int sin_force_idx_center = 3*24*numAnglesPerPeptide*(peptide)+24*numAnglesPerPeptide+12*2*angleNum;
        int sin_force_idx_right = 3*24*numAnglesPerPeptide*(peptide+1)+12*2*angleNum;

        int cos_force_idx_left = 3*24*numAnglesPerPeptide*(peptide-1)+2*24*numAnglesPerPeptide+12*(2*angleNum+1);
        int cos_force_idx_center = 3*24*numAnglesPerPeptide*(peptide)+24*numAnglesPerPeptide+12*(2*angleNum+1);
        int cos_force_idx_right = 3*24*numAnglesPerPeptide*(peptide+1)+12*(2*angleNum+1);

        real3 sinForce;
        real3 cosForce;

        for (int j=0;j<4;j++){
            sinForce = cosAngle * angle_force[j];
            if (peptide > 0) {
                angleForceArray[sin_force_idx_left+3*j] = sinForce.x;
                angleForceArray[sin_force_idx_left+3*j+1] = sinForce.y;
                angleForceArray[sin_force_idx_left+3*j+2] = sinForce.z;
            }
            angleForceArray[sin_force_idx_center+3*j] = sinForce.x;
            angleForceArray[sin_force_idx_center+3*j+1] = sinForce.y;
            angleForceArray[sin_force_idx_center+3*j+2] = sinForce.z;
            if (peptide < numPeptides-1) {
                angleForceArray[sin_force_idx_right+3*j] = sinForce.x;
                angleForceArray[sin_force_idx_right+3*j+1] = sinForce.y;
                angleForceArray[sin_force_idx_right+3*j+2] = sinForce.z;
            }
        }
        for (int j=0;j<4;j++){
            cosForce = -sinAngle * angle_force[j];
            if (peptide > 0) {
                angleForceArray[cos_force_idx_left+3*j] = cosForce.x;
                angleForceArray[cos_force_idx_left+3*j+1] = cosForce.y;
                angleForceArray[cos_force_idx_left+3*j+2] = cosForce.z;
            }
            angleForceArray[cos_force_idx_center+3*j] = cosForce.x;
            angleForceArray[cos_force_idx_center+3*j+1] = cosForce.y;
            angleForceArray[cos_force_idx_center+3*j+2] = cosForce.z;
            if (peptide < numPeptides-1) {
                angleForceArray[cos_force_idx_right+3*j] = cosForce.x;
                angleForceArray[cos_force_idx_right+3*j+1] = cosForce.y;
                angleForceArray[cos_force_idx_right+3*j+2] = cosForce.z;
            }
        }
    }
}

extern "C" __global__
void accumulateParticleForces(const real* __restrict__ NapShiftForces,
                              const real* __restrict__ angleForces,
                              const int* __restrict__ perParticleForceIndices,
                              const int* __restrict__ NapShiftParticleIndices,
                              int maxForcesPerParticle,
                              int numNapShiftParticles,
                              const int* __restrict__ indexToAtom,
                              long long* __restrict__ forceBuffers,
                              int paddedNumAtoms) {  
    for (int particleNum = blockIdx.x*blockDim.x+threadIdx.x; particleNum < numNapShiftParticles; particleNum += blockDim.x*gridDim.x) {
        int particle = NapShiftParticleIndices[particleNum];
        int atom = indexToAtom[particle];
        int forceIndex = 0;
        real perParticleForceX = 0;
        real perParticleForceY = 0;
        real perParticleForceZ = 0;
        for (int i=0; i < maxForcesPerParticle; i++) {
            forceIndex = perParticleForceIndices[particleNum*maxForcesPerParticle + i];
            if (forceIndex < 0) {
                break;
            }
            perParticleForceX += (NapShiftForces[3*forceIndex] * angleForces[3*forceIndex]);
            perParticleForceY += (NapShiftForces[3*forceIndex+1] * angleForces[3*forceIndex+1]);
            perParticleForceZ += (NapShiftForces[3*forceIndex+2] * angleForces[3*forceIndex+2]);
        }
        forceBuffers[atom] += (long long) (perParticleForceX*0x100000000);
        forceBuffers[atom+paddedNumAtoms] += (long long) (perParticleForceY*0x100000000);
        forceBuffers[atom+2*paddedNumAtoms] += (long long) (perParticleForceZ*0x100000000);
    }
}

extern "C" __global__
void swapAtomToIndex(const int* __restrict__ atomToIndex, const int numAtoms, int* __restrict__ indexToAtom) {
    int k;
    for (int atom = blockIdx.x*blockDim.x+threadIdx.x; atom < numAtoms; atom += blockDim.x*gridDim.x) {
        k = atomToIndex[atom];
        indexToAtom[k] = atom;
    }
}

extern "C" __global__
void DownloadCSDifferenceAvgData(real* __restrict__ avgCSTensor,
                                 const real* downloadedData,
                                 int numNapShiftParticles) {
    for (int particle = blockIdx.x*blockDim.x+ threadIdx.x; particle < numNapShiftParticles; particle += blockDim.x*gridDim.x) {
        avgCSTensor[6*particle+0] = downloadedData[6*particle+0];
        avgCSTensor[6*particle+1] = downloadedData[6*particle+1];
        avgCSTensor[6*particle+2] = downloadedData[6*particle+2];
        avgCSTensor[6*particle+3] = downloadedData[6*particle+3];
        avgCSTensor[6*particle+4] = downloadedData[6*particle+4];
        avgCSTensor[6*particle+5] = downloadedData[6*particle+5];
    }
}