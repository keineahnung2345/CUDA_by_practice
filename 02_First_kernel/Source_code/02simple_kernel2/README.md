# Notes
## void**
```cpp
int *d_c;
const int C_BYTES = 1 * sizeof(int);
cudaMalloc( (void**)&d_c, C_BYTES );
```

From [how to use void ** pointer correctly?](https://stackoverflow.com/questions/9040818/how-to-use-void-pointer-correctly), `void**` is a pointer to pointer.
`cudaMalloc` use `void**` pointer to change the value(aka, where it points to) of `int*` pointer `d_c`.
