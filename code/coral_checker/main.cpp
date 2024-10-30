#include <iostream>
#include <cstdio>
#include <string>
#include <regex>
#include <cstdlib>

int main() {
    FILE* pipe = popen("lsusb", "r");
    if (!pipe) {
        std::cerr << "Failed to run lsusb command." << std::endl;
        return 1;
    }

    std::regex device_regex(R"(Bus (\d{3}) Device (\d{3}):.*Global Unichip Corp\.)");
    char buffer[128];
    std::string bus, device;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        std::smatch match;

        if (std::regex_search(line, match, device_regex)) {
            bus = match[1];
            device = match[2];
            std::cout << "Found device on Bus " << bus << " and Device " << device << std::endl;
            break;
        }
    }

    pclose(pipe);
                             
    if (!bus.empty() && !device.empty()) {
        std::string chmod_command = "sudo chmod 666 /dev/bus/usb/" + bus + "/" + device;
        std::cout << "Running command: " << chmod_command << std::endl;

        int result = system(chmod_command.c_str());
        if (result == 0) {
            std::cout << "Permissions updated successfully." << std::endl;
        } else {
            std::cerr << "Failed to update permissions." << std::endl;
        }
    } else {
        std::cerr << "Device 'Global Unichip Corp.' not found." << std::endl;
    }

    return 0;
}
