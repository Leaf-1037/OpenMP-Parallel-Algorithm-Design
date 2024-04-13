#include <iostream>
#include <omp.h> // OpenMP编程需要包含的头文件
#define N 1000000
#define num_of_threads 16

using namespace std;

double a[N];
double S[N];

int main()
{
    double tt;
    for (int i = 0; i < N; ++i)
    {
        a[i] = (double)i;
        /*std::cout << sum << std::endl;*/
    }
    tt = omp_get_wtime();
    S[0] = a[0];
    for (int i = 1; i < N; ++i) {
        S[i] = S[i - 1] + a[i];
    }
    tt = omp_get_wtime() - tt;
    cout << "Comparison: Sequential Time = " << tt << "s. (Serial)" << endl << endl;



    for (int threads = 1; threads <= 16; ++threads) {
        double ts = 0.0;
        for (int grp = 0; grp < 10; ++grp) {
            memset(S, 0, sizeof S);
            omp_set_num_threads(num_of_threads);
            tt = omp_get_wtime();
#pragma omp parallel shared(S)
            {
                double partial_Sum = 0.0;
                int chunk = N / (num_of_threads - 1);
#pragma omp for nowait 
                for (int i = 0; i < num_of_threads; ++i) {
                    int j = chunk * i;
                    S[j] = a[j];
                    for (j++; j < min(N, chunk * (i + 1)); ++j) {
                        S[j] = S[j - 1] + a[j];
                    }
                }
#pragma omp barrier
#pragma omp single
                for (int i = 1; i * chunk + chunk - 1 < N; ++i) {
                    S[i * chunk + chunk - 1] += S[i * chunk - 1];
                }
#pragma omp for nowait 
                for (int i = 1; i < num_of_threads; ++i) {
                    int j;
                    for (j = chunk * i; j < min(N, chunk * (i + 1)); ++j) {
                        S[j] += S[i * chunk - 1];
                    }
                }

                
            }
            tt = omp_get_wtime() - tt;

            ts += tt;
        }
        cout << "# Thread_Num = " << threads << "; Time = " << ts / 10 << "s. (using Scan)" << endl;
    }
    

    

    return 0;
}