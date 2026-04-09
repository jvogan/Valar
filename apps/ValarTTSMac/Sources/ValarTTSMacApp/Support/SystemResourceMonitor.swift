import Foundation
import Darwin.Mach
import ValarSystemMemorySupport

enum SystemResourceMonitor {
    static func currentProcessFootprintBytes() -> Int {
        guard let taskVMInfoRev1Offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else {
            assertionFailure(
                "SystemResourceMonitor could not determine task_vm_info_data_t.min_address offset."
            )
            return 0
        }

        let taskVMInfoRev1Count = mach_msg_type_number_t(
            taskVMInfoRev1Offset / MemoryLayout<integer_t>.size
        )
        let taskVMInfoCount = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        var info = task_vm_info_data_t()
        var count = taskVMInfoCount

        let result = withUnsafeMutablePointer(to: &info) { infoPointer in
            infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { integerPointer in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), integerPointer, &count)
            }
        }

        guard result == KERN_SUCCESS else {
            assertionFailure("SystemResourceMonitor.task_info(TASK_VM_INFO) failed with status \(result).")
            return 0
        }

        guard count >= taskVMInfoRev1Count else {
            assertionFailure(
                "SystemResourceMonitor.task_info(TASK_VM_INFO) returned \(count) words; expected at least \(taskVMInfoRev1Count)."
            )
            return 0
        }

        return Int(clamping: info.phys_footprint)
    }

    static func availableMemoryBytes() -> Int {
        Int(clamping: valar_os_proc_available_memory())
    }

    static func diskUsageBytes(at path: String?) -> Int {
        guard let path, !path.isEmpty else { return 0 }

        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }

        if let directFileSize = allocatedSize(for: url) {
            return directFileSize
        }

        let keys: [URLResourceKey] = [
            .isRegularFileKey,
            .totalFileAllocatedSizeKey,
            .fileAllocatedSizeKey,
            .totalFileSizeKey,
            .fileSizeKey,
        ]
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total = 0
        for case let fileURL as URL in enumerator {
            total += allocatedSize(for: fileURL) ?? 0
        }
        return total
    }

    private static func allocatedSize(for url: URL) -> Int? {
        guard let values = try? url.resourceValues(forKeys: [
            .isRegularFileKey,
            .totalFileAllocatedSizeKey,
            .fileAllocatedSizeKey,
            .totalFileSizeKey,
            .fileSizeKey,
        ]) else {
            return nil
        }

        guard values.isRegularFile == true else {
            return nil
        }

        return values.totalFileAllocatedSize
            ?? values.fileAllocatedSize
            ?? values.totalFileSize
            ?? values.fileSize
    }
}
