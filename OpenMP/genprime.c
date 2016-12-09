#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

int main(int argc, char **argv){
  if (argc != 3){
    printf("Usage: genprime N t\n");
    exit(1);
  }
  
  int n = atoi(argv[1]);
  if (n <= 2){
    printf("N should be larger than 2!\n");
    exit(1);
  }
  
  int t = atoi(argv[2]);
  if (t <= 0){
    printf("t should be a positive integer");
    exit(1);
  }
  int pl[n];
  double tstart = 0.0, ttaken;

  tstart = omp_get_wtime();
  omp_set_num_threads(t);
 
#pragma omp parallel
  {
    int i, j, p;

#pragma omp for schedule(static, (n-1)/t)
    // Initialize the array
    for (i=0; i<n-1; i++){
      pl[i] = i+2;
    }
   
#pragma omp for schedule(static, 1) 
    //main loop
    for (i=0; i<(n+1)/2-1; i++){    
      j = 2;
      if (pl[i]!= 0){
	while (((p = pl[i]) != 0) && (p*j <= n)){
	  pl[p*j-2] = 0;	
	  j += 1;
	}
      }
    }
  }
  ttaken = omp_get_wtime() - tstart;
  printf("Time take for the main part: %f\n", ttaken);

  // Write to file
  int fp;
  if ((fp = open ("N.txt", O_CREAT|O_WRONLY|O_TRUNC, 0777)) < 0)
    printf("Error open the file for writing: %s\n", strerror(errno));

  char to_write[200000];
  char line[30];
  int counter, last_p, e, i;
  
  counter = 1;
  last_p = 2;
  
  for (i=0; i<n-1; i++){
    if (pl[i] != 0){
      sprintf(line, "%d, %d, %d\n",counter, pl[i], pl[i]-last_p);      
      last_p = pl[i];
      strcat(to_write, line);
      counter += 1;
    }
  }
  if ((e = write(fp, to_write, strlen(to_write))) < 0)
    printf("Error write to file: %s\n", strerror(errno));
  
  close(fp);
  return 0;
}
  
