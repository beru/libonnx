#ifndef __HMAP_H__
#define __HMAP_H__

#include <onnxconf.h>

#include "list.h"

#ifdef __cplusplus
extern "C" {
#endif

struct hmap_t {
	hlist_head * hash;
	list_head list;
	unsigned int size;
	unsigned int n;
};

struct hmap_entry_t {
	hlist_node node;
	list_head head;
	char * key;
	void * value;
};

#define hmap_for_each_entry(entry, m) \
	list_for_each_entry(entry, &(m)->list, head)

#define hmap_for_each_entry_reverse(entry, m) \
	list_for_each_entry_reverse(entry, &(m)->list, head)

hmap_t * hmap_alloc(unsigned int size);
void hmap_free(hmap_t * m, void (*cb)(hmap_entry_t *));
void hmap_clear(hmap_t * m, void (*cb)(hmap_entry_t *));
void hmap_add(hmap_t * m, const char * key, void * value);
void hmap_remove(hmap_t * m, const char * key);
void hmap_sort(hmap_t * m);
void * hmap_search(hmap_t * m, const char * key);

#ifdef __cplusplus
}
#endif

#endif /* __HMAP_H__ */
