#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor t = torch::randn({2, 3});
  std::cout << t << std::endl;
  return 0;
}
