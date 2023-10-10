#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double generate(int m, int n) 
{
    srand(time(NULL));
    
    int i, j;
    int* x = (int*)malloc(m * n * sizeof(int));
    double* y = (double*)malloc(m * sizeof(double));

    // x is an even distribution of 1s and 7s
    for (i = 0; i < m; i++) 
    {
        for (j = 0; j < n; j++) 
        {
            x[i * n + j] = (rand() % 2 == 0) ? 1 : 7;
        }
    }

    // Calculate y values
    for (i = 0; i < m; i++) 
    {
        double sum = 0.0;
        for (j = 0; j < n; j++) 
        {
            sum += x[i * n + j];
        }
        y[i] = sum / n;
    }

    // Calculate randvar
    double randvar = 0.0;
    for (i = 0; i < m; i++) 
    {
        randvar += ((y[i] - 4) * (y[i] - 4)) / n;
    }

    free(x);
    free(y);

    return randvar / m;
}

int main() {
    int m = 100;
    int x_size = 100;
    int x_step = (10000 - 100) / (x_size - 1);
    int* x = (int*)malloc(x_size * sizeof(int));

    // Generate x values
    for (int i = 0; i < x_size; i++) 
    {
        x[i] = 100 + i * x_step;
    }


    // Calculate theoretical expectation
    double* expectation_theo = (double*)malloc(x_size * sizeof(double));
    for (int i = 0; i < x_size; i++) 
    {
        expectation_theo[i] = 9.0 / (x[i] * x[i]);
    }

    // Calculate simulated expectation
    double* expectation_sim = (double*)malloc(x_size * sizeof(double));
    for (int i = 0; i < x_size; i++) 
    {
        expectation_sim[i] = generate(m, x[i]);
    }

    // Save data to a .dat file
    FILE *fp = fopen("output.dat", "w");
    if (fp == NULL) 
    {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    for (int i = 0; i < x_size; i++) 
    {
        fprintf(fp, "%d\t%.15lf\t%.15lf\n", x[i], expectation_theo[i], expectation_sim[i]);
    }

    fclose(fp);

    // Free allocated memory
    free(x);
    free(expectation_theo);
    free(expectation_sim);

    return 0;
}