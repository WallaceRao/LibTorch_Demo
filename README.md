# LibTorch_Demo
A simply demo about using libtorch

How to build
1 modify CMakeList.txt, change CMAKE_PREFIX_PATH to your libtorch path
2 cmake .
3 make

How to Run
1 python model.py to export jit file
2 ./demo test.jit

Output
the terminal output should like:

      (base) root@n227-011-046:~/torch_test/cpp_test# ./demo test.jit
      output1 shape:[1, 120]
      output2 shape:[1, 84]
      output3 shape:[1, 10]
      output4 value:100
      ok

Explanation
The torch script has a function named "forward", the function returns a tuple which contains 4 elements, the first 3 elements are tensors, and the last one is an integer.
In main.cc, we can use module.run_method to get the tuple, name the tuple as "result", and we can get the first tensor by 
result.toTuple()->elements()[0].toTensor()
Similarly, we can get the last integer by result.toTuple()->elements()[3].toInt()


