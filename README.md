# ssNetMF
Network embedding approximation algorithm. Linear complexity in time and space. High Performance.

## Requirement
MKL

## argv
```C
long long i;
window_size = atoi(argv[2]);
strcpy(filename, argv[4]);
strcpy(outputname, argv[6]);
n = atoi(argv[8]);
nnz = atoll(argv[10]);
b = atoi(argv[12]);
h = atoi(argv[14]);
dim = atoi(argv[16]);
batch = atoi(argv[18]);
q = atoi(argv[20]);
s1 = atoi(argv[22]);
s2 = atoi(argv[24]);
s3 = atoi(argv[26]);
use_freigs_convex = atoi(argv[28]);
alpha = atof(argv[30]);
```

## data format
(%d, %d) %lf

"(1, 2) 3" means Node(1) -> Node(2) and the edge value is 3.