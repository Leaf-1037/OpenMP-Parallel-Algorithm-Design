#include <iostream>
#include <omp.h> // OpenMP编程需要包含的头文件
#define N 10000000
#define num_of_threads 16

using namespace std;

int main()
{
    double sum = 0.0;
    double tt;
    tt = omp_get_wtime();
    for (int i = 0; i < N; ++i)
    {
        sum = sum + (double)i;
        /*std::cout << sum << std::endl;*/
    }
    tt = omp_get_wtime() - tt;
    cout << "Time = " << tt << "s. (Serial)" << endl;
    std::cout << "Before: " << sum << std::endl;

//    for (int threads = 1; threads <= 16; ++threads) {
//        double ts = 0.0;
//        for (int grp = 0; grp < 10; ++grp) {
//            sum = 0.0;
//            omp_set_num_threads(num_of_threads);
//            tt = omp_get_wtime();
//#pragma omp parallel for reduction(+: sum) 
//            for (int i = 0; i < N; ++i)
//            {
//                sum = sum + (double)i;
//                /*std::cout << sum << std::endl;*/
//            }
//            tt = omp_get_wtime() - tt;
//
//            ts += tt;
//        }
//        cout << "#"<<threads<<"Time = " << ts/10 << "s. (using reduction(+:sum))" << endl;
//    }

    for (int threads = 1; threads <= 16; ++threads) {
        double ts = 0.0;
        for (int grp = 0; grp < 10; ++grp) {
            double S = 0.0;
            omp_set_num_threads(num_of_threads);
            tt = omp_get_wtime();
#pragma omp parallel shared(S)
            {
                double partial_Sum = 0.0;
#pragma omp for nowait
                for (int i = 0; i < N; ++i) {
                    partial_Sum += (double)i;
                    /*int ss = omp_get_thread_num();
                    cout << "Thread #" << ss << " is executing." <<" LS = "<<partial_Sum<< endl;*/
                }
                S += partial_Sum;
            }
            tt = omp_get_wtime() - tt;

            ts += tt;
        }
        cout << "#" << threads << "Time = " << ts / 10 << "s. (using partial_Sum)" << endl;
    }
    

    

    return 0;
}