#include "ValarSystemMemorySupport.h"

#include <mach/mach.h>
#include <mach/mach_host.h>

uint64_t valar_os_proc_available_memory(void) {
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        mach_host_self(),
        HOST_VM_INFO64,
        (host_info64_t)&vm_stat,
        &count
    );
    if (kr != KERN_SUCCESS) {
        return 0;
    }
    uint64_t free_pages = (uint64_t)vm_stat.free_count
                        + (uint64_t)vm_stat.inactive_count
                        + (uint64_t)vm_stat.purgeable_count;
    return free_pages * (uint64_t)vm_page_size;
}
