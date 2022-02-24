//
// Created by devilox on 30.04.2021.
//
//-----------------------------//
#include "AkulaPackage/SerialConnector.h"
//-----------------------------//
///---TODO: fix description---///
/**
 * @param tSerialPath
 * @param tSpeed
 * @param tTimeout For some unknown reason tty timeout is in deciseconds (10^-2s) but tTimeout will be in millisecond
 */
SerialConnector::SerialConnector(const std::string& tSerialPath, uint16_t tSpeed, uint32_t tTimeout, size_t tPacketSize) {
    mTimeoutMs      = tTimeout;
    mSerialPort     = open(tSerialPath.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);

    if (mSerialPort < 0) {
        std::cerr << "Failed to open the port: " << errno << "(" + std::string(strerror(errno)) + ")" << std::endl;
        throw std::runtime_error("SerialConnector::constructor");
    }

    //----------//

    if (tcgetattr(mSerialPort, &mTTY) != 0) {
        std::cerr << "Failed to get attributes: " << errno << "(" + std::string(strerror(errno)) + ")" << std::endl;
        throw std::runtime_error("SerialConnector::constructor");
    }

    mTTY.c_cflag &= ~PARENB;                                                // Clear parity bit, disabling parity (most common)
    mTTY.c_cflag &= ~CSTOPB;                                                // Clear stop field, only one stop bit used in communication (most common)
    mTTY.c_cflag &= ~CSIZE;                                                 // Clear all bits that set the data size
    mTTY.c_cflag |= CS8;                                                    // 8 bits per byte (most common)
    mTTY.c_cflag &= ~CRTSCTS;                                               // Disable RTS/CTS hardware flow control (most common)
    mTTY.c_cflag |= CREAD | CLOCAL;                                         // Turn on READ & ignore ctrl lines (CLOCAL = 1)

    mTTY.c_lflag &= ~ICANON;
    mTTY.c_lflag &= ~ECHO;                                                  // Disable echo
    mTTY.c_lflag &= ~ECHOE;                                                 // Disable erasure
    mTTY.c_lflag &= ~ECHONL;                                                // Disable new-line echo
    mTTY.c_lflag &= ~ISIG;                                                  // Disable interpretation of INTR, QUIT and SUSP
    mTTY.c_iflag &= ~(IXON | IXOFF | IXANY);                                // Turn off s/w flow ctrl
    mTTY.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);       // Disable any special handling of received bytes

    mTTY.c_oflag &= ~OPOST;                                                 // Prevent special interpretation of output bytes (e.g. newline chars)
    mTTY.c_oflag &= ~ONLCR;                                                 // Prevent conversion of newline to carriage return/line feed

    mTTY.c_cc[VTIME] = 0;                                                   // Polling time interval
    mTTY.c_cc[VMIN] = 0;

    cfsetispeed(&mTTY, tSpeed);
    cfsetospeed(&mTTY, tSpeed);

    if (tcsetattr(mSerialPort, TCSANOW, &mTTY) != 0) {
        std::cerr << "Failed to set attributes: " << errno << "(" + std::string(strerror(errno)) + ")" << std::endl;
        throw std::runtime_error("SerialConnector::constructor");
    }

    fcntl(mSerialPort, F_SETFL, FNDELAY);

    //----------//

    mPacketSize     = tPacketSize;
}
SerialConnector::~SerialConnector() {
    close(mSerialPort);
}
//-----------------------------//
ssize_t SerialConnector::readSerial(uint8_t* tBuffer) const {
    size_t BytesTotalRead = 0;
    ssize_t BytesRead;
    int Available;

//    while (BytesTotalRead < mPacketSize) {
        if (checkNewData()) {
            if (ioctl(mSerialPort, TIOCINQ, &Available) != -1) {
                std::cout << Available << std::endl;

                BytesRead = read(mSerialPort, tBuffer + BytesTotalRead, mPacketSize - BytesTotalRead);

                if (BytesRead < 1) {
                    std::cerr << "readSerial: write smth" << std::endl;
                    return 0;
                } else {
                    BytesTotalRead += BytesRead;
                }
            } else {
                std::cerr << "readSerial: write smth" << std::endl;
                return -1;
            }
        }
//    }

    return BytesTotalRead;
}
ssize_t SerialConnector::writeSerial(const uint8_t* tBuffer) const {
    return write(mSerialPort, tBuffer, mPacketSize);
}
//-----------------------------//
bool SerialConnector::checkNewData() const {
    fd_set  ReadFDs;
    int Count;
    timespec Timeout {
            .tv_sec     = mTimeoutMs / 1000,
            .tv_nsec    = mTimeoutMs % 1000 * 1'000'000
    };

    FD_ZERO(&ReadFDs);
    FD_SET(mSerialPort, &ReadFDs);

    Count = pselect(mSerialPort + 1, &ReadFDs, nullptr, nullptr, &Timeout, nullptr);

    if (Count > 0) {
        return true;
    } else {
        switch (Count) {
            case 0:
//                std::cout << "Timeout" << std::endl;
                return false;
            case -1:
                std::cerr << "Throw exception or smth" << std::endl;
                return false;
            default:
                std::cerr << "It's bad or idk" << std::endl;
                return false;
        }
    }
}