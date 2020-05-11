#include "OfflineOffset.h"
#include <cassert>

const void *OfflineOffset::arenaBase = 0;
size_t OfflineOffset::arenaLen = 0;
const void *OfflineOffset::fbBase = 0;
size_t OfflineOffset::fbLen = 0;

void OfflineOffset::Init(void *arenaPtr, size_t arenaSz,
                         const std::vector<char> &fb) {
  arenaBase = arenaPtr;
  arenaLen = arenaSz;
  fbBase = fb.data();
  fbLen = fb.size();
}

OfflineOffset::OfflineOffset(const void *p) {
  assert(arenaBase && "OfflineOffset not initialized!");

  set(p);
}

void OfflineOffset::set(const void *p) {
  if (!p) {
    m_type = Type::Null;
  } else if (p >= arenaBase && p < ((char *)arenaBase + arenaLen)) {
    m_type = Type::Arena;
    m_offset = (uintptr_t)p - (uintptr_t)arenaBase;
  } else if (p >= fbBase && p < ((char *)fbBase + fbLen)) {
    m_type = Type::FB;
    m_offset = (uintptr_t)p - (uintptr_t)fbBase;
  } else {
    assert(false && "OfflineOffset: Pointer must be in buffer that will be "
                    "on the target!");
  }
}

std::string OfflineOffset::getPtrCode() const {
  switch (m_type) {
  case Type::Null:
    return "nullptr";
  case Type::Arena:
    return "(tensor_arena + " + std::to_string(m_offset) + ")";
  case Type::FB:
    return "(g_model_data + " + std::to_string(m_offset) + ")";
  }
}
