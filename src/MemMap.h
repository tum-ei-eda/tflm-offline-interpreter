#ifndef OFFLINE_INTERPRETER_MEMMAP_H
#define OFFLINE_INTERPRETER_MEMMAP_H

#include <string>
#include <vector>

#include "OfflineOffset.h"

// Keeps track of Arena and Flatbuffer buffers and prints a summary.
class MemMap {
 public:
  void record(OfflineOffset offset, size_t len, const std::string &tag);
  void report() const;

 private:
  struct Entry {
    int base;
    size_t len;
    std::string tag;
  };
  std::vector<Entry> m_constEntries;
  std::vector<Entry> m_arenaEntries;
};

#endif
