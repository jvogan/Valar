import CoreAudio
import Foundation

struct AudioOutputDevice: Identifiable, Equatable {
    static let systemDefaultID = "system-default"

    let id: String
    let name: String
    let isSystemDefault: Bool

    static let systemDefault = AudioOutputDevice(
        id: systemDefaultID,
        name: "System Default",
        isSystemDefault: true
    )
}

enum SystemAudioDeviceCatalog {
    static func outputDevices() -> [AudioOutputDevice] {
        let defaultDeviceID = defaultOutputDeviceID()

        let resolvedDevices = deviceIDs()
            .filter(hasOutputChannels)
            .compactMap { deviceID -> AudioOutputDevice? in
                guard let uid = stringProperty(
                    selector: kAudioDevicePropertyDeviceUID,
                    scope: kAudioObjectPropertyScopeGlobal,
                    objectID: deviceID
                ), let name = stringProperty(
                    selector: kAudioObjectPropertyName,
                    scope: kAudioObjectPropertyScopeGlobal,
                    objectID: deviceID
                ) else {
                    return nil
                }

                return AudioOutputDevice(
                    id: uid,
                    name: name,
                    isSystemDefault: deviceID == defaultDeviceID
                )
            }
            .sorted { lhs, rhs in
                if lhs.isSystemDefault != rhs.isSystemDefault {
                    return lhs.isSystemDefault
                }
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }

        return [AudioOutputDevice.systemDefault] + resolvedDevices
    }

    private static func deviceIDs() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var propertySize: UInt32 = 0

        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propertySize
        ) == noErr else {
            return []
        }

        let count = Int(propertySize) / MemoryLayout<AudioDeviceID>.size
        var ids = Array(repeating: AudioDeviceID(), count: count)

        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propertySize,
            &ids
        ) == noErr else {
            return []
        }

        return ids
    }

    private static func defaultOutputDeviceID() -> AudioDeviceID? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID()
        var propertySize = UInt32(MemoryLayout<AudioDeviceID>.size)

        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propertySize,
            &deviceID
        )

        return status == noErr ? deviceID : nil
    }

    private static func hasOutputChannels(_ deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )
        var propertySize: UInt32 = 0

        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &propertySize) == noErr else {
            return false
        }

        let bufferListStorage = UnsafeMutableRawPointer.allocate(
            byteCount: Int(propertySize),
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { bufferListStorage.deallocate() }

        guard AudioObjectGetPropertyData(
            deviceID,
            &address,
            0,
            nil,
            &propertySize,
            bufferListStorage
        ) == noErr else {
            return false
        }

        let bufferList = bufferListStorage.assumingMemoryBound(to: AudioBufferList.self)
        let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
        return buffers.contains { $0.mNumberChannels > 0 }
    }

    private static func stringProperty(
        selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope,
        objectID: AudioObjectID
    ) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var propertySize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var value: Unmanaged<CFString>?

        let status = withUnsafeMutablePointer(to: &value) { valuePointer in
            AudioObjectGetPropertyData(
                objectID,
                &address,
                0,
                nil,
                &propertySize,
                valuePointer
            )
        }

        guard status == noErr, let value else {
            return nil
        }

        return value.takeRetainedValue() as String
    }
}
