#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: demo <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  auto result = module.run_method("forward", torch::ones({1, 100})); // result is a tuple contains 3 tensors and one integer
  auto first_output = result.toTuple()->elements()[0].toTensor();
  auto second_output = result.toTuple()->elements()[1].toTensor();
  auto third_output = result.toTuple()->elements()[2].toTensor();
  auto forth_output = result.toTuple()->elements()[3].toInt();
  std::cout<<"output1 shape:"<<first_output.sizes()<< std::endl;
  std::cout<<"output2 shape:"<<second_output.sizes()<< std::endl;
  std::cout<<"output3 shape:"<<third_output.sizes()<< std::endl;
  std::cout<<"output4 value:"<<forth_output<< std::endl;

  std::cout << "ok\n";
}
