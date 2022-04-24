TEMPLATE = """
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <math.h> 

#define index2D(i, j, N)  ((i)*(N)) + (j)
#define index3D(i, j, k, N)  ((i)*(N)*(N)) + ((j)*(N)) + (k)
{gpu}
{cpu}
int main(int argc, char * argv[])
{{
  unsigned long N;
  unsigned int b1;
  unsigned int b2;
  unsigned int b3;
  unsigned int g1;
  unsigned int g2;
  unsigned int g3;
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 8)
  {{
    fprintf(stderr, "usage: dummy num b1 b2 b3 g1 g2 g3\\n");
    fprintf(stderr, "num = size of each dimesnion\\n");
    fprintf(stderr, "b1 = size of block dim 1\\n");
    fprintf(stderr, "b2 = size of block dim 2\\n");
    fprintf(stderr, "b3 = size of block dim 3\\n");
    fprintf(stderr, "g1 = size of block dim 1\\n");
    fprintf(stderr, "g2 = size of block dim 2\\n");
    fprintf(stderr, "g3 = size of block dim 3\\n");
    exit(1);
  }}
  
  N = (unsigned long) atoi(argv[1]);
  b1 = (unsigned int) atoi(argv[2]);
  b2 = (unsigned int) atoi(argv[3]);
  b3 = (unsigned int) atoi(argv[4]);
  g1 = (unsigned int) atoi(argv[5]);
  g2 = (unsigned int) atoi(argv[6]);
  g3 = (unsigned int) atoi(argv[7]);
  
  /* Dynamically allocate array of floats for CPU*/
  {init_cpu}
  {malloc_cpu}

  /* Dynamically allocate array of floats for GPU*/
  {init_gpu}
  {malloc_gpu}

  /* Assign random values */
  for (int i = 0; i < {size}; i++){{
    {random_init}
  }}

  start = clock();
  func_cpu({cpu_call}, N);
  end = clock();

  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("CPU=%lf\\n", time_taken);

  dim3 block(b1, b2, b3);
  dim3 grid(g1, g2, g3);

  start = clock();
  func_gpu({gpu_call}, N, block, grid, {size}*sizeof(float)); 
  end = clock();    
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  printf("GPU=%lf\\n", time_taken);

  int bad_count = 0;
  for (int i = 0; i < {size}; i++){{
      {compare}
  }}
  printf("BAD=%d\\n", bad_count);

  {free_cpu}
  {free_gpu}

  return 0;

}}

"""