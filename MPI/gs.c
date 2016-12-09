#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
int pass = 0;
int partial_sz, sp;
float ** local_coef;

/****** Function declarations */

int check_err_rate(float * new_x); /* Check to see if the error rate for new Xs passes */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
void get_new_x(float * new_partial_x, int my_rank); /* Calculate a partial set of new Xs */
void parse_msg(float * recvd_msg, int my_rank);

/********************************/

/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

int check_err_rate(float * new_x){
  int i;
  float cur_err;
  for (i=0; i<num; i++){
    cur_err = fabs((new_x[i]-x[i]) / new_x[i]);
    if (cur_err > err)
      return 0;
  }
  return 1;
 }

/************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

  fscanf(fp,"%d ",&num);
  fscanf(fp,"%f ",&err);

  /* Now, time to allocate the matrices and vectors */
  a = (float**)malloc(num * sizeof(float*));
  if(!a)
    {
      printf("Cannot allocate a!\n");
      exit(1);
    }

  for(i = 0; i < num; i++) 
    {
      a[i] = (float *)malloc(num * sizeof(float)); 
      if(!a[i])
  	{
	  printf("Cannot allocate a[%d]!\n",i);
	  exit(1);
  	}
    }
 
  x = (float *) malloc(num * sizeof(float));
  if(!x)
    {
      printf("Cannot allocate x!\n");
      exit(1);
    }

  b = (float *) malloc(num * sizeof(float));
  if(!b)
    {
      printf("Cannot allocate b!\n");
      exit(1);
    }

  /* The initial values of Xs */
  for(i = 0; i < num; i++)
    fscanf(fp,"%f ", &x[i]);
 
  for(i = 0; i < num; i++)
    {
      for(j = 0; j < num; j++)
	fscanf(fp,"%f ",&a[i][j]);
   
      /* reading the b element */
      fscanf(fp,"%f ",&b[i]);
    }
 
  fclose(fp); 
}

/************************************************************/
void get_new_x(float * new_partial_x, int my_rank){
  int i,j;
  for (i=0; i<partial_sz; i++){
    new_partial_x[i] = b[i];
    for (j=0; j<num; j++){
      if (j != my_rank*sp+i)
	new_partial_x[i] -= (local_coef[i][j] * x[j]);
    }
    new_partial_x[i] /= local_coef[i][my_rank*sp+i];
  }
}

/************************************************************/
void parse_msg(float * recvd_msg, int my_rank){
  float * p = recvd_msg;
  int i,j;

  num = (int) (*p);
  p++;
  sp = (int) (*p);
  p++;
  partial_sz = (int) (*p);
  p++;
  err = *p;
  p++;

  if (my_rank != 0){
    x = malloc(num * sizeof(float));
    b = malloc(partial_sz * sizeof(float));
  }
  memcpy(x, p, num*sizeof(float));
  p += num;

  memcpy(b, p, partial_sz*sizeof(float));
  p += partial_sz;
  
  local_coef = malloc(partial_sz * sizeof(float*));
  for (i=0; i<partial_sz; i++)
    local_coef[i] = malloc(num * sizeof(float));
 
  for (i=0; i<partial_sz; i++){    
    for (j=0; j<num; j++){
      local_coef[i][j] = *p;
      p++;
    }
  }
}

/************************************************************/


int main(int argc, char *argv[])
{

  int i,j, actual_proc, cur;
  int my_rank, comm_sz;
  int nit = 0; /* number of iterations */
  int send_count = 0;
  float *local_arr, *p;
  float recv_buf[4096];

/* Start up MPI and get related configs */
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  if(argc != 2){
    printf("Usage: gsref filename\n");
    exit(1);
  }

  if (my_rank == 0){
    /* Read the input file and fill the global data structure above */ 
    get_input(argv[1]);
    /* Check for convergence condition */
    check_matrix();

    /* In case #unknowns < #processes*/
    if (num >= comm_sz){
      actual_proc = comm_sz;
      partial_sz = (num + comm_sz - 1) / comm_sz;
    }
    else{
      partial_sz = 1;
      actual_proc = num;
    }
   
    sp = partial_sz; // saved partial_sz
   
    /* Construct the send_buf that contains: 
       #unknowns, general #x no corner case, #x for each process, error rate, initial x value, 
       constants for each process, and coefficients for each processes */
    float * send_buf[comm_sz];
    for (i=0; i<comm_sz; i++)
      send_buf[i] = malloc(4096 * sizeof(float));

    for (i=0; i<comm_sz; i++){

      if (partial_sz * (i+1) > num)
	partial_sz = num - partial_sz*i;
      if (i >= actual_proc || partial_sz < 0)
	partial_sz = 0;

      //      printf("partial_sz for rank %d is : %d\n",i, partial_sz);

      p = send_buf[i];
      *p = (float) num; // #unknowns
      p++;
    
      *p = (float) sp; // general #x for each process negelect corner cases
      p++;
	
      *p = (float) partial_sz; // #x for each process
      p++;

      *p = err; //error rate
      p++;

      memcpy(p, x, num*sizeof(float)); //initial x
      p += num;      
     
      memmove(p, b+sp*i, partial_sz*sizeof(float)); //const
      p += partial_sz;
            
      send_count = 4 + num + partial_sz;

      for (j=sp*i; j<sp*i+partial_sz; j++){
	memmove(p, a[j], num*sizeof(float));
	p += num;
	send_count += num;
      } //coefficient
      
      if (i == 0)
	parse_msg(send_buf[i], my_rank);
      else
	MPI_Send(send_buf[i], send_count, MPI_FLOAT, i, i, MPI_COMM_WORLD);     
      partial_sz = sp; // set partial_sz for rank 0 back to its real value;
    }

  }
  else{
      MPI_Recv(recv_buf, 4096, MPI_FLOAT, 0, my_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      parse_msg(recv_buf, my_rank);
  }

  float new_xs[num];
  float new_partial_x[sp];

  while (!pass){
    get_new_x(new_partial_x, my_rank);
    
    MPI_Allgather(new_partial_x, sp, MPI_FLOAT, new_xs, sp, MPI_FLOAT, MPI_COMM_WORLD);

    nit++;
    pass = check_err_rate(new_xs);

    for (i=0; i<num; i++)
      x[i] = new_xs[i];
  }

   
  /* Writing to the stdout */
    /* Keep that same format */
  if (my_rank == 0){
    for(i = 0; i < num; i++)
      printf("%f\n", x[i]);
    printf("total number of iterations: %d\n", nit);
  }

  MPI_Finalize();
  exit(0);
}
