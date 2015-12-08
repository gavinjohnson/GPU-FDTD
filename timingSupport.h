//
//  Copyright [Dan Connors: Dan.Connors@gmail.com]
//

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define MAX_PERF_SAMPLES 1000

//-------------------------------------
// Timing Functions
//-------------------------------------
typedef struct perfTimer
{
    int sampleIndex;
    struct timeval start[MAX_PERF_SAMPLES];
    struct timeval stop[MAX_PERF_SAMPLES];
    double time[MAX_PERF_SAMPLES];
} perfTimer;

// Performance functions
void initTimer(perfTimer *perf)
{
    int i;
    perf->sampleIndex = 0;
    for (i=0; i < MAX_PERF_SAMPLES; i++) {
        perf->time[i] = 0;
    }
}

void startTimer(perfTimer *perf)
{
    int sample;
    sample = perf->sampleIndex;
    gettimeofday(&(perf->start[sample]),NULL);
}

void stopTimer(perfTimer *perf)
{
    struct timeval diff;
    int sample;
    
    // Get the stop time
    sample = perf->sampleIndex;
    gettimeofday(&(perf->stop[sample]),NULL);
    
    // Subtract the times;
    timersub(&(perf->stop[sample]), &(perf->start[sample]), &diff);
    
    // Convert to seconds
    double time = diff.tv_sec + diff.tv_usec/1000000.0;
    
    // Record time
    perf->time[sample] = time;
    
    // Move on to the next sample (assume pair of calls: start, stop)
    perf->sampleIndex++;
}

float returnTimerSample(perfTimer *perf, int index)
{
    if ((index >= 0)&&(index < MAX_PERF_SAMPLES)) {
        return perf->time[index];
    }
    return (-1);
}

void printTimerSample(perfTimer *perf, int index)
{
    if ((index >= 0)&&(index < MAX_PERF_SAMPLES)) {
        printf(" %f \n", perf->time[index]);
    }
}

void printTimerAll(const char *timerName, perfTimer *perf)
{
    int index;
    printf("%s : \n",timerName);
    for (index=0; index < perf->sampleIndex; index++) {
        printf("  %d : ",index);
        printTimerSample(perf, index);
    }
}

void printSpeedupTimer(perfTimer *tbase, perfTimer *topti, int index)
{
    printf("%f", tbase->time[index] / topti->time[index]);
}

