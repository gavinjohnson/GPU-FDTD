/* Copyright (C) Gavin Johnson, 2015
 * Feel free do distribute, modify,
 * and use this code however you
 * would like.  Just leave this
 * notice at the top.  Enjoy!
 *
 * gavin.johnson@outlook.com
 *
 */



//Includes
#include <libc.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>
#include <math.h>
#include <time.h>
#include "timingSupport.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Macro value to string - used to pass the nodes to the openCL kernel
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// GPU computation?
#ifndef GPUCOMP
#define GPUCOMP 1
#endif

// array indexing macros
#define Ex(i,j,k) (Ex[(i) + NX*(j) + NX*NY*(k)])
#define Ey(i,j,k) (Ey[(i) + NX*(j) + NX*NY*(k)])
#define Ez(i,j,k) (Ez[(i) + NX*(j) + NX*NY*(k)])
#define Hx(i,j,k) (Hx[(i) + NX*(j) + NX*NY*(k)])
#define Hy(i,j,k) (Hy[(i) + NX*(j) + NX*NY*(k)])
#define Hz(i,j,k) (Hz[(i) + NX*(j) + NX*NY*(k)])
#define field_access(i,j,k) (field[(i) + NX*(j) + NX*NY*(k)])
// Number of nodes in each direction
#ifndef NX
#define NX 40
#endif
#ifndef NY
#define NY 40
#endif
#ifndef NZ
#define NZ 40

#endif
// Macro for passing the nodes to the openCL kernel
#define NDEF "-DNX=" STR(NX) " -DNY=" STR(NY) " -DNZ=" STR(NZ)

#define iSIG 2
#define jSIG 2
#define kSIG 2


// the OpenCL source
static char * programSource = "                                                                                          \n"\
"// array indexing macros                                                                                                \n"\
"#define Ex(i,j,k) (Ex[(i) + NX*(j) + NX*NY*(k)])                                                                        \n"\
"#define Ey(i,j,k) (Ey[(i) + NX*(j) + NX*NY*(k)])                                                                        \n"\
"#define Ez(i,j,k) (Ez[(i) + NX*(j) + NX*NY*(k)])                                                                        \n"\
"#define Hx(i,j,k) (Hx[(i) + NX*(j) + NX*NY*(k)])                                                                        \n"\
"#define Hy(i,j,k) (Hy[(i) + NX*(j) + NX*NY*(k)])                                                                        \n"\
"#define Hz(i,j,k) (Hz[(i) + NX*(j) + NX*NY*(k)])                                                                        \n"\
"#ifndef NX                                                                                                              \n"\
"#define NX 11                                                                                                           \n"\
"#endif                                                                                                                  \n"\
"#ifndef NY                                                                                                              \n"\
"#define NY 11                                                                                                           \n"\
"#endif                                                                                                                  \n"\
"#ifndef NZ                                                                                                              \n"\
"#define NZ 11                                                                                                           \n"\
"#endif                                                                                                                  \n"\
"                                                                                                                        \n"\
"__kernel void Eupdate( __global float * Ex,__global float * Ey, __global float * Ez,__global float * Hx,__global float * Hy,__global float * Hz, float cExy, float cExz, float cEyz, float cEyx, float cEzx, float cEzy)\n"\
"{                                                                                                                       \n"\
"// set up the indices                                                                                                   \n"\
"size_t i,j,k;                                                                                                           \n"\
"i = get_global_id(0);                                                                                                   \n"\
"j = get_global_id(1);                                                                                                   \n"\
"k = get_global_id(2);                                                                                                   \n"\
"//Ex Update                                                                                                             \n"\
"if (i<NX-1 & j<NY-1 & k<NZ-1 & k>=1 & j>=1)                                                                             \n"\
"{                                                                                                                       \n"\
"Ex(i,j,k) = Ex(i,j,k) + cExy*(Hz(i,j,k)-Hz(i,j-1,k)) - cExz*(Hy(i,j,k)-Hy(i,j,k-1));                                    \n"\
"}                                                                                                                       \n"\
"//Ey Update                                                                                                             \n"\
"if (i<NX-1 & j<NY-1 & k<NZ-1 & k>=1 & i>=1)                                                                             \n"\
"{                                                                                                                       \n"\
"Ey(i,j,k) = Ey(i,j,k) + cEyz*(Hx(i,j,k)-Hx(i,j,k-1)) - cEyx*(Hz(i,j,k)-Hz(i-1,j,k));                                    \n"\
"}                                                                                                                       \n"\
"//Ez Update                                                                                                             \n"\
"if (i<NX-1 & j<NY-1 & k<NZ-1 & j>=1 & i>=1)                                                                             \n"\
"{                                                                                                                       \n"\
"Ez(i,j,k) = Ez(i,j,k) + cEzx*(Hy(i,j,k)-Hy(i-1,j,k)) - cEzy*(Hx(i,j,k)-Hx(i,j-1,k));                                    \n"\
"}                                                                                                                       \n"\
"                                                                                                                        \n"\
"}                                                                                                                       \n"\
"                                                                                                                        \n"\
"                                                                                                                        \n"\
"__kernel void Hupdate( __global float * Ex,__global float * Ey,__global float * Ez, __global float * Hx,__global float * Hy,__global float * Hz, float cHxy, float cHxz, float cHyz, float cHyx, float cHzx, float cHzy)\n"\
"{                                                                                                                       \n"\
"// set up indices                                                                                                       \n"\
"size_t i,j,k;                                                                                                           \n"\
"i = get_global_id(0);                                                                                                   \n"\
"j = get_global_id(1);                                                                                                   \n"\
"k = get_global_id(2);                                                                                                   \n"\
"//Hx Update                                                                                                             \n"\
"if (i<NX & j<NY-1 & k<NZ-1)                                                                                             \n"\
"{                                                                                                                       \n"\
"Hx(i,j,k) = Hx(i,j,k) - cHxy*(Ez(i,j+1,k)-Ez(i,j,k)) + cHxz*(Ey(i,j,k+1)-Ey(i,j,k));                                    \n"\
"}                                                                                                                       \n"\
"//Hy Update                                                                                                             \n"\
"if (i<NX-1 & j<NY & k<NZ-1)                                                                                             \n"\
"{                                                                                                                       \n"\
"Hy(i,j,k) = Hy(i,j,k) - cHyz*(Ex(i,j,k+1)-Ex(i,j,k)) + cHyx*(Ez(i+1,j,k)-Ez(i,j,k));                                    \n"\
"}                                                                                                                       \n"\
"//Hz Update                                                                                                             \n"\
"if (i<NX-1 & j<NY-1 & k<NZ)                                                                                             \n"\
"{                                                                                                                       \n"\
"Hz(i,j,k) = Hz(i,j,k) - cHzx*(Ey(i+1,j,k)-Ey(i,j,k)) + cHzy*(Ex(i,j+1,k)-Ex(i,j,k));                                    \n"\
"}                                                                                                                       \n"\
"}                                                                                                                       \n"\
"                                                                                                                        \n";



// initialize the fields
void initialize(float * field, int * size){
    for (int i = 0; i < *size; i++){
        field[i] = 0;
    }
}

// update the source
void srcUpdate(float * field, float * t, float * coef){
    // differentiated gaussian
    static float tw = 1e-9;         // standard deviation of pulse
    static float to = 5 * 1e-9;     // mean time of pulse
    static float mag = 1;           // source magnitude
    field_access(iSIG,jSIG,kSIG) += (*coef)*(-(mag * exp(-pow(((*t) - to),2) / pow(tw,2)) * (2.0*(*t) - 2.0*to))/tw);
}


// update the E-field (non GPU)
void Eupdate_serial(float * Ex, float * Ey, float * Ez, float * Hx, float * Hy, float * Hz,
                    float * cExy, float * cExz, float * cEyz, float * cEyx, float * cEzx, float * cEzy){
    int i,j,k;
    const float c_cExy = *cExy;
    const float c_cExz = *cExy;
    const float c_cEyz = *cEyz;
    const float c_cEyx = *cEyx;
    const float c_cEzx = *cEzx;
    const float c_cEzy = *cEzy;
    
    
    // Ex update
    for (k=1;k<NZ-1;k++){
        for (j=1;j<NY-1;j++){
            for (i=0;i<NX-1;i++){
                Ex(i,j,k) = Ex(i,j,k) + c_cExy*(Hz(i,j,k)-Hz(i,j-1,k)) - c_cExz*(Hy(i,j,k)-Hy(i,j,k-1));
            }
        }
    }
    
    // Ey update
    for (k=1;k<NZ-1;k++){
        for (j=0;j<NY-1;j++){
            for (i=1;i<NX-1;i++){
                Ey(i,j,k) = Ey(i,j,k) + c_cEyz*(Hx(i,j,k)-Hx(i,j,k-1)) - c_cEyx*(Hz(i,j,k)-Hz(i-1,j,k));
            }
        }
    }
    
    // Ez update
    for (k=0;k<NZ-1;k++){
        for (j=1;j<NY-1;j++){
            for (i=1;i<NX-1;i++){
                Ez(i,j,k) = Ez(i,j,k) + c_cEzx*(Hy(i,j,k)-Hy(i-1,j,k)) - c_cEzy*(Hx(i,j,k)-Hx(i,j-1,k));
            }
        }
    }
}

// update the H-field (non GPU)
void Hupdate_serial(float * Ex, float * Ey, float * Ez, float * Hx, float * Hy, float * Hz,
                    float * cHxy, float * cHxz, float * cHyz, float * cHyx, float * cHzx, float * cHzy){
    int i,j,k;
    const float c_cHxy = *cHxy;
    const float c_cHxz = *cHxy;
    const float c_cHyz = *cHyz;
    const float c_cHyx = *cHyx;
    const float c_cHzx = *cHzx;
    const float c_cHzy = *cHzy;
    
    // Hx update
    for (k=0;k<NZ-1;k++){
        for (j=0;j<NY-1;j++){
            for (i=0;i<NX;i++){
                Hx(i,j,k) = Hx(i,j,k) - c_cHxy*(Ez(i,j+1,k)-Ez(i,j,k)) + c_cHxz*(Ey(i,j,k+1)-Ey(i,j,k));            }
        }
    }
    // Hy update
    for (k=0;k<NZ-1;k++){
        for (j=0;j<NY;j++){
            for (i=0;i<NX-1;i++){
                Hy(i,j,k) = Hy(i,j,k) - c_cHyz*(Ex(i,j,k+1)-Ex(i,j,k)) + c_cHyx*(Ez(i+1,j,k)-Ez(i,j,k));
            }
        }
    }
    // Hz update
    for (k=0;k<NZ;k++){
        for (j=0;j<NY-1;j++){
            for (i=0;i<NX-1;i++){
                Hz(i,j,k) = Hz(i,j,k) - c_cHzx*(Ey(i+1,j,k)-Ey(i,j,k)) + c_cHzy*(Ex(i,j+1,k)-Ex(i,j,k));
            }
        }
    }
}



int main()
{
    cl_context context;
    cl_context_properties properties[3];
    cl_kernel Eupdate;
    cl_kernel Hupdate;
    cl_command_queue command_queue;
    cl_program program;
    cl_int err;
    cl_uint num_of_platforms=0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices=0;
    cl_mem cEx, cEy, cEz, cHx, cHy, cHz;
    cl_mem cSampleVec;
    
    // set up the workgroup size (next value divisible by 8)
    size_t globalSize[3] = {NX,NY,NZ};
    
    // Courant Number
    double CFLN = 0.99;
    
    // Simulation time
    double simtime = 4e-7;
    
    // material parameters and speed of light
    double mu_o = 1.2566370614e-6;
    double eps_o = 8.854187817e-12;
    double mu_r = 1;
    double eps_r = 1;
    double mu= mu_r * mu_o;
    double eps = eps_r * eps_o;
    double c = 1.0/sqrt(mu*eps);
    
    // Global Domain Size
    double Dx = 1;
    double Dy = 1;
    double Dz = 1;
    
    // discretization size computation
    double dx=Dx/((double)NX-1);
    double dy=Dy/((double)NY-1);
    double dz=Dz/((double)NZ-1);
    double dt=CFLN/(c*sqrt(pow(dx,-2)+pow(dy,-2)+pow(dz,-2)));
    float cJ = dt/eps;
    
    cl_int nt = floor(simtime/dt);
    
    // sampling vector initialization
    int sampleBytes = nt*sizeof(cl_float);
    cl_float * sampleVec = (float *)malloc(sampleBytes);
    for (int i=0;i<sampleBytes;i++)
    {
        sampleVec[i]=0;
    }
    
    // time variable
    cl_float t = 0;
    
    // size of the spaces
    cl_int sizEx = NX*NY*NZ;
    cl_int sizEy = NX*NY*NZ;
    cl_int sizEz = NX*NY*NZ;
    cl_int sizHx = NX*NY*NZ;
    cl_int sizHy = NX*NY*NZ;
    cl_int sizHz = NX*NY*NZ;
    
    // build coefficient matrices
    // E-field coefficients
    cl_float cExy = (dt/(eps*dy));
    cl_float cExz = (dt/(eps*dz));
    cl_float cEyx = (dt/(eps*dx));
    cl_float cEyz = (dt/(eps*dz));
    cl_float cEzx = (dt/(eps*dx));
    cl_float cEzy = (dt/(eps*dy));
    // H-field coefficients
    cl_float cHxy = (dt/(mu*dy));
    cl_float cHxz = (dt/(mu*dz));
    cl_float cHyx = (dt/(mu*dx));
    cl_float cHyz = (dt/(mu*dz));
    cl_float cHzx = (dt/(mu*dx));
    cl_float cHzy = (dt/(mu*dy));
    
    // Set up the local memory
    cl_float * Ex;
    cl_float * Ey;
    cl_float * Ez;
    cl_float * Hx;
    cl_float * Hy;
    cl_float * Hz;
    
    //retreive a list of platforms avaible
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms)!= CL_SUCCESS)
    {
        printf("Unable to get platform_id\n");
        return 1;
    }
    
    // try to get a supported GPU device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
    {
        printf("Unable to get device_id\n");
        return 1;
    }
    
    // context properties list - must be terminated with 0
    properties[0]= CL_CONTEXT_PLATFORM;
    properties[1]= (cl_context_properties) platform_id;
    properties[2]= 0;
    
    //create a context with the GPU device
    context = clCreateContext(properties,1,&device_id,NULL,NULL,&err);
    //printf("error check clCreateContext: %d\n",err);
    
    // create command queue using the context and device
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    //printf("error check clCreateCommandQueue: %d\n",err);
    
    // create a program from the kernel source code (above)
    program = clCreateProgramWithSource(context,1,(const char **) &programSource, NULL, &err);
    //printf("error check clCreateProgramWithSource: %d\n",err);
    
    // compile the program
    //if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
    if (clBuildProgram(program, 0, NULL, NDEF, NULL, NULL) != CL_SUCCESS)
    {
        printf("error check clBuildProgram: %d\n",err);
        printf("Error building program\n");
        return 1;
    }
    
    // specify which kernels from the program to execute
    Eupdate = clCreateKernel(program, "Eupdate", &err);
    //printf("error check clCreateKernel: %d\n",err);
    Hupdate = clCreateKernel(program, "Hupdate", &err);
    //printf("error check clCreateKernel: %d\n",err);
    
    // Create the input and output arrays in device memory for our calculation
    cEx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizEx*sizeof(cl_float), NULL, &err);
    cEy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizEy*sizeof(cl_float), NULL, &err);
    cEz = clCreateBuffer(context, CL_MEM_READ_WRITE, sizEz*sizeof(cl_float), NULL, &err);
    cHx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizHx*sizeof(cl_float), NULL, &err);
    cHy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizHy*sizeof(cl_float), NULL, &err);
    cHz = clCreateBuffer(context, CL_MEM_READ_WRITE, sizHz*sizeof(cl_float), NULL, &err);
    cSampleVec = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sampleBytes, NULL, &err);
    //printf("error check clCreateBuffer: %d\n",err);
    
    // point to the memory space
    Ex = (cl_float*)clEnqueueMapBuffer(command_queue, cEx, CL_TRUE, CL_MAP_WRITE, 0, sizEx*sizeof(cl_float), 0, NULL, NULL, &err);
    Ey = (cl_float*)clEnqueueMapBuffer(command_queue, cEy, CL_TRUE, CL_MAP_WRITE, 0, sizEy*sizeof(cl_float), 0, NULL, NULL, &err);
    Ez = (cl_float*)clEnqueueMapBuffer(command_queue, cEz, CL_TRUE, CL_MAP_WRITE, 0, sizEz*sizeof(cl_float), 0, NULL, NULL, &err);
    Hx = (cl_float*)clEnqueueMapBuffer(command_queue, cHx, CL_TRUE, CL_MAP_WRITE, 0, sizHx*sizeof(cl_float), 0, NULL, NULL, &err);
    Hy = (cl_float*)clEnqueueMapBuffer(command_queue, cHy, CL_TRUE, CL_MAP_WRITE, 0, sizHy*sizeof(cl_float), 0, NULL, NULL, &err);
    Hz = (cl_float*)clEnqueueMapBuffer(command_queue, cHz, CL_TRUE, CL_MAP_WRITE, 0, sizHz*sizeof(cl_float), 0, NULL, NULL, &err);
    sampleVec = (cl_float*)clEnqueueMapBuffer(command_queue, cSampleVec, CL_TRUE, CL_MAP_WRITE, 0, sampleBytes, 0, NULL, NULL, &err);
    
    initialize(Ex, &sizEx);
    initialize(Ey, &sizEy);
    initialize(Ez, &sizEz);
    initialize(Hx, &sizHx);
    initialize(Hy, &sizHy);
    initialize(Hz, &sizHz);
    
    
    // Set E-kernel args
    // E args: (Ex,Ey,Ez,Hx,Hy,Hz,cExy,cExz,cEyz,cEyx,cEzx,cEzy)
    err  = clSetKernelArg(Eupdate, 0, sizeof(cl_mem), &cEx);
    err |= clSetKernelArg(Eupdate, 1, sizeof(cl_mem), &cEy);
    err |= clSetKernelArg(Eupdate, 2, sizeof(cl_mem), &cEz);
    err |= clSetKernelArg(Eupdate, 3, sizeof(cl_mem), &cHx);
    err |= clSetKernelArg(Eupdate, 4, sizeof(cl_mem), &cHy);
    err |= clSetKernelArg(Eupdate, 5, sizeof(cl_mem), &cHz);
    err |= clSetKernelArg(Eupdate, 6, sizeof(cl_float), &cExy);
    err |= clSetKernelArg(Eupdate, 7, sizeof(cl_float), &cExz);
    err |= clSetKernelArg(Eupdate, 8, sizeof(cl_float), &cEyz);
    err |= clSetKernelArg(Eupdate, 9, sizeof(cl_float), &cEyx);
    err |= clSetKernelArg(Eupdate, 10, sizeof(cl_float), &cEzx);
    err |= clSetKernelArg(Eupdate, 11, sizeof(cl_float), &cEzy);
    
    // Set H-kernel args
    // H args: (Ex,Ey,Ez,Hx,Hy,Hz,cHxy,cHxz,cHyz,cHyx,cHzx,cHzy)
    err |= clSetKernelArg(Hupdate, 0, sizeof(cl_mem), &cEx);
    err |= clSetKernelArg(Hupdate, 1, sizeof(cl_mem), &cEy);
    err |= clSetKernelArg(Hupdate, 2, sizeof(cl_mem), &cEz);
    err |= clSetKernelArg(Hupdate, 3, sizeof(cl_mem), &cHx);
    err |= clSetKernelArg(Hupdate, 4, sizeof(cl_mem), &cHy);
    err |= clSetKernelArg(Hupdate, 5, sizeof(cl_mem), &cHz);
    err |= clSetKernelArg(Hupdate, 6, sizeof(cl_float), &cHxy);
    err |= clSetKernelArg(Hupdate, 7, sizeof(cl_float), &cHxz);
    err |= clSetKernelArg(Hupdate, 8, sizeof(cl_float), &cHyz);
    err |= clSetKernelArg(Hupdate, 9, sizeof(cl_float), &cHyx);
    err |= clSetKernelArg(Hupdate, 10, sizeof(cl_float), &cHzx);
    err |= clSetKernelArg(Hupdate, 11, sizeof(cl_float), &cHzy);
    //printf("error check clSetKernelArg: %d\n",err);
    
    
    // Write our data set into the input array in device memory
    //     clEnqueueWriteBuffer(commandQueue, exd, CL_TRUE, 0, exSize * sizeof(cl_float), ex, 0, NULL, NULL);
    err  = clEnqueueWriteBuffer(command_queue, cEx, CL_TRUE, 0, sizEx*sizeof(cl_float), Ex, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, cEy, CL_TRUE, 0, sizEy*sizeof(cl_float), Ey, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, cEz, CL_TRUE, 0, sizEz*sizeof(cl_float), Ez, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, cHx, CL_TRUE, 0, sizHx*sizeof(cl_float), Hx, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, cHy, CL_TRUE, 0, sizHy*sizeof(cl_float), Hy, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, cHz, CL_TRUE, 0, sizHz*sizeof(cl_float), Hz, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(command_queue, cSampleVec, CL_TRUE, 0, sampleBytes, sampleVec, 0, NULL, NULL);
    //printf("error check clEnqueueWriteBuffer: %d\n",err);
    
    // timer init and start
    float toc;
    perfTimer timer;
    initTimer(&timer);
    startTimer(&timer);
    //************************************  Run the Kernel  ************************************//
    //nt=1000;
    if (GPUCOMP) { // run on GPU
        printf("Starting GPU Kernel Execution...\n");
        for (cl_int i=0;i<=nt;i++){
            
            // execute the E-field updates
            err = clEnqueueNDRangeKernel(command_queue, Eupdate, 3, NULL, globalSize, NULL,0, NULL, NULL);//localSize
            
            // Wait for the command queue to get serviced before reading back results
            clFinish(command_queue);
            
            // increment time
            t += dt;
            
            // execute the H-field updates
            err = clEnqueueNDRangeKernel(command_queue, Hupdate, 3, NULL, globalSize, NULL,0, NULL, NULL);//localSize
            
            // Wait for the command queue to get serviced before reading back results
            clFinish(command_queue);
            
            // Get samples from near the center of the space
            sampleVec[i] = Ex((int)(NX/2+1),(int)(NY/2+1),(int)(NZ/2+1));//Ex(3,3,3);//Ex((int)(NX/2+1),(int)(NY/2+1),(int)(NZ/2+1));
            
            // update the source
            srcUpdate(Ex, &t, &cJ);
        }
    }
    else{  // run on CPU
        printf("Starting CPU Kernel Execution...\n");
        for (cl_int i=0;i<=nt;i++){
            
            // execute the E-field updates
            Eupdate_serial(Ex,Ey,Ez,Hx,Hy,Hz,&cExy,&cExz,&cEyz,&cEyx,&cEzx,&cEzy);
            
            // increment time
            t += dt;
            
            // execute the H-field updates
            Hupdate_serial(Ex,Ey,Ez,Hx,Hy,Hz,&cHxy,&cHxz,&cHyz,&cHyx,&cHzx,&cHzy);
            
            
            // Get samples from near the center of the space
            sampleVec[i] = Ex((int)(NX/2+1),(int)(NY/2+1),(int)(NZ/2+1));
            
            // update the source
            srcUpdate(Ex, &t, &cJ);
            
        }
        
    }
    // stop timer and print results
    stopTimer(&timer);
    toc = returnTimerSample(&timer, 0);
    printf("...Execution Complete\nTotal Execution Time : %f\n",toc);
    // get the results
    FILE * fp_params = fopen("params.log","w");
    FILE * fp_samps = fopen("sample.log", "w");
    fprintf(fp_params,"dt:%.10e\n",dt);
    fprintf(fp_params,"c:%.10e\n",c);
    fprintf(fp_params,"t:%.10e\n",toc);
    fprintf(fp_params,"N:%d\n",NX);
    fprintf(fp_params,"GPU:%d\n",GPUCOMP);
    for (int i = 0; i < nt; i++) {
        //if (sampleVec[i] != 0.0){ printf("%d : %f\n",i,sampleVec[i]);}
        fprintf(fp_samps,"%.10e\n",sampleVec[i]);
        //printf("%.10e\n",sampleVec[i]);
    }
    fclose(fp_params);
    fclose(fp_samps);
    // free the shared memory
    clReleaseMemObject(cEx);
    clReleaseMemObject(cEy);
    clReleaseMemObject(cEz);
    clReleaseMemObject(cHx);
    clReleaseMemObject(cHy);
    clReleaseMemObject(cHz);
    clReleaseMemObject(cSampleVec);
    return 0;
}

