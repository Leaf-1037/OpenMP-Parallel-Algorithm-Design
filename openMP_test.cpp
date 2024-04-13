#include<omp.h>

#include<iostream>

int main()

{

    std::cout << "parallel begin:\n";

#pragma omp parallel

    {

        std::cout << omp_get_thread_num();

    }

    std::cout << "\n parallel end.\n";

    std::cin.get();

    return 0;

}
