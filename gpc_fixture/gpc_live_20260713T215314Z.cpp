// GPC live integration fixture. Safe to delete after audit retention.
#include <string>
#include <vector>

namespace gpc_fixture {

constexpr char kRunMarker[] = "gpc-live-20260713T215314Z";

int sumPositive(const std::vector<int>& values) {
  int total = 0;
  for (int value : values) {
    if (value > 0) {
      total = total + value;
    }
  }
  return total;
}

std::string stateLabel(bool ready) {
  return ready ? "ready" : "waiting";
}

} // namespace gpc_fixture
