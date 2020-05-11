#include "OptimalMemPlanner.h"

TfLiteStatus OptimalMemPlanner::AddBuffer(tflite::ErrorReporter *error_reporter,
                                          int size, int first_time_used,
                                          int last_time_used) {
  m_bufferInfo.push_back(
      {(int)m_bufferInfo.size(), size, first_time_used, last_time_used});
  m_needCalc = true;
}

size_t OptimalMemPlanner::GetMaximumMemorySize() {
  CalcIfNeeded();
  return 0;
}

int OptimalMemPlanner::GetBufferCount() { return 0; }

TfLiteStatus
OptimalMemPlanner::GetOffsetForBuffer(tflite::ErrorReporter *error_reporter,
                                      int buffer_index, int *offset) {
  CalcIfNeeded();
  return kTfLiteOk;
}

class AdjacencyMatrix {
public:
  AdjacencyMatrix(int numElems)
      : m_buf(numElems * numElems), m_numElems(numElems) {}

  void setAdj(int i, int j) {
    m_buf[i * m_numElems + j] = 1;
    m_buf[j * m_numElems + i] = 1;
  }
  bool isAdj(int i, int j) const { return m_buf[i * m_numElems + j] != 0; }

private:
  std::vector<int> m_buf;
  int m_numElems;
};

void OptimalMemPlanner::CalcIfNeeded() {
  // Brute force plan: Try every allocation combination

  // Get upper bound of time.
  auto itMaxTime = std::max_element(
      m_bufferInfo.begin(), m_bufferInfo.end(),
      [](const auto &a, const auto &b) { return a.last_use < b.last_use; });
  if (itMaxTime == m_bufferInfo.end()) {
    // No buffers.
    return;
  }

  // Map time slots to buffers used at that time.
  std::vector<std::vector<const BufferInfo *>> timeSlots(itMaxTime->last_use);
  for (int i = 0; i < timeSlots.size(); i++) {
    for (const auto &info : m_bufferInfo) {
      if (info.first_use <= i && info.last_use >= i) {
        timeSlots[i].push_back(&info);
      }
    }
  }

  // Build adjecancy matrix.
  AdjacencyMatrix adjMat(m_bufferInfo.size());
  for (const auto &info : m_bufferInfo) {
    adjMat.setAdj(0, 0);
  }
}
