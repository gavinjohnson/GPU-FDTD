// array indexing macros
#define Ex(i,j,k) (Ex[(i) + NX*(j) + NX*NY*(k)])
#define Ey(i,j,k) (Ey[(i) + NX*(j) + NX*NY*(k)])
#define Ez(i,j,k) (Ez[(i) + NX*(j) + NX*NY*(k)])
#define Hx(i,j,k) (Hx[(i) + NX*(j) + NX*NY*(k)])
#define Hy(i,j,k) (Hy[(i) + NX*(j) + NX*NY*(k)])
#define Hz(i,j,k) (Hz[(i) + NX*(j) + NX*NY*(k)])
#ifndef NX
#define NX 11
#endif
#ifndef NY
#define NY 11
#endif
#ifndef NZ
#define NZ 11
#endif

__kernel void Eupdate( __global float * Ex,__global float * Ey, __global float * Ez,__global float * Hx,__global float * Hy,__global float * Hz, float cExy, float cExz, float cEyz, float cEyx, float cEzx, float cEzy)
{
    // set up the indices
    size_t i,j,k;
    i = get_global_id(0);
    j = get_global_id(1);
    k = get_global_id(2);
    //Ex Update
    if (i<NX-1 & j<NY-1 & k<NZ-1 & k>=1 & j>=1)
    {
        Ex(i,j,k) = Ex(i,j,k) + cExy*(Hz(i,j,k)-Hz(i,j-1,k)) - cExz*(Hy(i,j,k)-Hy(i,j,k-1));
    }
    //Ey Update
    if (i<NX-1 & j<NY-1 & k<NZ-1 & k>=1 & i>=1)
    {
        Ey(i,j,k) = Ey(i,j,k) + cEyz*(Hx(i,j,k)-Hx(i,j,k-1)) - cEyx*(Hz(i,j,k)-Hz(i-1,j,k));
    }
    //Ez Update
    if (i<NX-1 & j<NY-1 & k<NZ-1 & j>=1 & i>=1)
    {
        Ez(i,j,k) = Ez(i,j,k) + cEzx*(Hy(i,j,k)-Hy(i-1,j,k)) - cEzy*(Hx(i,j,k)-Hx(i,j-1,k));
    }
    
}

__kernel void Hupdate( __global float * Ex,__global float * Ey,__global float * Ez, __global float * Hx,__global float * Hy,__global float * Hz, float cHxy, float cHxz, float cHyz, float cHyx, float cHzx, float cHzy)
{
    // set up indices
    size_t i,j,k;
    i = get_global_id(0);
    j = get_global_id(1);
    k = get_global_id(2);
    //Hx Update
    if (i<NX & j<NY-1 & k<NZ-1)
    {
        Hx(i,j,k) = Hx(i,j,k) - cHxy*(Ez(i,j+1,k)-Ez(i,j,k)) + cHxz*(Ey(i,j,k+1)-Ey(i,j,k));
    }
    //Hy Update
    if (i<NX-1 & j<NY & k<NZ-1)
    {
        Hy(i,j,k) = Hy(i,j,k) - cHyz*(Ex(i,j,k+1)-Ex(i,j,k)) + cHyx*(Ez(i+1,j,k)-Ez(i,j,k));
    }
    //Hz Update
    if (i<NX-1 & j<NY-1 & k<NZ)
    {
        Hz(i,j,k) = Hz(i,j,k) - cHzx*(Ey(i+1,j,k)-Ey(i,j,k)) + cHzy*(Ex(i,j+1,k)-Ex(i,j,k));
    }
}