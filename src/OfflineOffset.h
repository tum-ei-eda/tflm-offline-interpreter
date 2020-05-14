#ifndef OFFLINE_INTERPRETER_OFFLINEOFFSET_H
#define OFFLINE_INTERPRETER_OFFLINEOFFSET_H

#include <string>
#include <vector>

// Converts offsets between the offline interpreter and the target buffers.
class OfflineOffset {
 public:
  enum class Type { Null, Arena, FB };

  static void Init(void *arenaPtr, size_t arenaSz, const std::vector<char> &fb);

  // p: Pointer inside of the offline interpreter.
  explicit OfflineOffset(const void *p);
  void set(const void *p);

  // Returns a code snippet that accesses the correct pointer on the target.
  std::string getPtrCode() const;

  Type getType() const { return m_type; }

 private:
  static const void *arenaBase;
  static size_t arenaLen;
  static const void *fbBase;
  static size_t fbLen;

  uintptr_t m_offset = 0;
  Type m_type = Type::Null;
};

#endif
